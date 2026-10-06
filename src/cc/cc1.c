#include "../config.h"

#include <ctype.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // FPC

#include "codegen.h"
#include "emit_code.h"
#include "fe_misc.h"
#include "lexer.h"
#include "parser.h"
#include "type.h"
#include "util.h"
#include "var.h"

////////////////////////////////////////////////

extern void install_builtins(Vector *decls);

static void init_compiler(Vector *decls, FILE *ofp) {
  init_lexer();
  init_global();
  init_emit(ofp);

#if XCC_TARGET_PROGRAMMING_MODEL == XCC_PROGRAMMING_MODEL_LP64
  // Default
#elif XCC_TARGET_PROGRAMMING_MODEL == XCC_PROGRAMMING_MODEL_ILP32
  set_fixnum_size(FX_CHAR,  1, 1);
  set_fixnum_size(FX_SHORT, 2, 2);
  set_fixnum_size(FX_INT,   4, 4);
  set_fixnum_size(FX_LONG,  4, 4);
  set_fixnum_size(FX_LLONG, 8, 8);
  set_fixnum_size(FX_ENUM,  4, 4);
#else
# error "Unsupported programming model"
#endif

  install_builtins(decls);
}

static void compile1(FILE *ifp, const char *filename, Vector *decls) {
  set_source_file(ifp, filename);
  parse(decls);
}

static void usage(FILE *fp) {
  fprintf(
      fp,
      "Usage: cc1 [options] file...\n"
      "Options:\n"
      "  -O<level>           (ignored)\n"
  );
}

int main(int argc, char *argv[]) {
  struct timespec start_ts;  // FPC
  clock_gettime(CLOCK_MONOTONIC, &start_ts);
  FILE *ofp = stdout;
  // FPC: -o FILE (the output: a driver in the same process can't take stdout)
  for (int i = 1; i + 1 < argc; ++i) {
    if (strcmp(argv[i], "-o") == 0) {
      ofp = fopen(argv[i + 1], "w");
      if (ofp == NULL)
        error("Cannot open output file: %s\n", argv[i + 1]);
      memmove(&argv[i], &argv[i + 2], (argc - i - 1) * sizeof(*argv));
      argc -= 2;
      break;
    }
  }

  enum {
    OPT_HELP = 128,
    OPT_VERSION,
    OPT_FNO,
    OPT_WNO,
    OPT_SSA,
  };

  static const struct option options[] = {
    {"-help", no_argument, OPT_HELP},
    {"v", no_argument, OPT_VERSION},
    {"-version", no_argument, OPT_VERSION},

    {"O", optional_argument},  // Optimization level

    // Sub command
    {"fno-", optional_argument, OPT_FNO},
    {"f", optional_argument},
    {"Wno-", optional_argument, OPT_WNO},
    {"W", required_argument},

    // Feature flag.
    {"-apply-ssa", no_argument, OPT_SSA},

    {NULL},
  };
  int opt;
  while ((opt = optparse(argc, argv, options)) != -1) {
    switch (opt) {
    default: assert(false); break;
    case OPT_HELP:
      usage(stdout);
      exit(0);
    case OPT_VERSION:
      show_version("cc1", XCC_TARGET_ARCH);
      return 0;

    case 'O':
      if (optarg == NULL) {
        cc_flags.optimize_level = 2;
      } else {
        char c = optarg[0];
        switch (c) {
        case '0':
        case '1':
        case '2':
        case '3':
        case 's':
        case 'z':
          cc_flags.optimize_level = isdigit(c) ? c - '0' : c;
          break;
        default:
          fprintf(stderr, "Warning: unknown optimization level: %c\n", c);
          break;
        }
      }
      break;

    case 'f':
    case OPT_FNO:
      if (optarg == NULL) {
        fprintf(stderr, "Warning: missing argument for -f\n");
        break;
      }
      if (!parse_fopt(optarg, opt == 'f')) {
        // Silently ignored.
        // fprintf(stderr, "Warning: unknown option for -f: %s\n", optarg);
      }
      break;

    case 'W':
      if (optarg != NULL) {
        if (strcmp(optarg, "error") == 0) {
          cc_flags.warn_as_error = true;
          break;
        }
        if (strcmp(optarg, "all") == 0) {
          // Assume all members are bool.
          for (bool *p = (bool*)&cc_flags.warn; p < (bool*)(&cc_flags.warn + 1); ++p)
            *p = true;
          break;
        }
      }
      // Fallthrough
    case OPT_WNO:
      if (optarg == NULL) {
        fprintf(stderr, "Warning: missing argument for -W\n");
        break;
      }
      if (strcmp(optarg, "error") == 0) {
        cc_flags.warn_as_error = true;
      } else if (!parse_wopt(optarg, opt == 'W')) {
        // Silently ignored.
        // fprintf(stderr, "Warning: unknown option for -W: %s\n", optarg);
      }
      break;

    case OPT_SSA:
      {
        extern bool apply_ssa;
        apply_ssa = true;
      }
      break;

    case '?':
      fprintf(stderr, "Warning: unknown option: %s\n", argv[optind - 1]);
      break;
    }
  }

  // Compile.
  Vector *toplevel = new_vector();
  init_compiler(toplevel, ofp);

  int iarg = optind;
  if (iarg >= argc)
    error("No input files");
  for (int i = iarg; i < argc; ++i) {
    const char *filename = argv[i];
    FILE *ifp;
    if (strcmp(filename, "-") == 0) {
      ifp = stdin;
      filename = "<stdin>";
    } else if (!is_file(filename) || (ifp = fopen(filename, "r")) == NULL) {
      error("Cannot open file: %s\n", filename);
    }
    compile1(ifp, filename, toplevel);
    if (ifp != stdin)
      fclose(ifp);
  }
  if (compile_error_count != 0)
    exit(1);
  if (cc_flags.warn_as_error && compile_warning_count != 0)
    exit(2);

  // FPC: XCC_PASS_TIME: the phases' times
  struct timespec ts0, ts1, ts2;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  gen(toplevel);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  emit_code(toplevel);
  clock_gettime(CLOCK_MONOTONIC, &ts2);
  if (getenv("XCC_PASS_TIME") != NULL) {
    fprintf(stderr, "cc1: parse %ld ms, gen %ld ms, emit %ld ms\n",
            (long)((ts0.tv_sec - start_ts.tv_sec) * 1000 + (ts0.tv_nsec - start_ts.tv_nsec) / 1000000),
            (long)((ts1.tv_sec - ts0.tv_sec) * 1000 + (ts1.tv_nsec - ts0.tv_nsec) / 1000000),
            (long)((ts2.tv_sec - ts1.tv_sec) * 1000 + (ts2.tv_nsec - ts1.tv_nsec) / 1000000));
  }

  if (diag_fp != NULL) {  // FPC: the warnings kept (cpp/pch.c)
    fclose(diag_fp);
    diag_fp = NULL;
    fwrite(diag_buf, 1, diag_size, stderr);
  }
  if (ofp != stdout)
    fclose(ofp);
  return 0;
}
