#include "../config.h"

#include <assert.h>
#include <string.h>

#include "pch.h"  // FPC
#include "preprocessor.h"
#include "util.h"

static void usage(FILE *fp) {
  fprintf(
      fp,
      "Usage: cpp [options] file...\n"
      "Options:\n"
      "  -D <label[=value]>  Define label\n"
      "  -I <path>           Add include path\n"
      "  -isystem <path>     Add system include path\n"
      "  -idirafter <path>   Add include path (lower priority)\n"
      "  -C                  Preserve comments\n"
      "  --pch=<dir>         Keep the preprocessed prefix in dir (FPC)\n"
      "  --no-prune          Every declaration of the system headers\n"
      "  --pch-key=<id>      The system headers' version: not checked\n"
  );
}

int main(int argc, char *argv[]) {
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
  init_preprocessor(ofp);

  enum {
    OPT_HELP = 128,
    OPT_VERSION,
    OPT_ISYSTEM,
    OPT_IDIRAFTER,
    OPT_PCH,  // FPC
    OPT_PCH_STATS,
    OPT_NO_PRUNE,
    OPT_PCH_KEY,
  };

  static const struct option options[] = {
    {"I", required_argument},  // Add include path
    {"isystem", required_argument, OPT_ISYSTEM},  // Add system include path
    {"idirafter", required_argument, OPT_IDIRAFTER},  // Add include path (after)
    {"D", required_argument},  // Define macro
    {"U", required_argument},  // Undefine macro
    {"C", no_argument},  // Do not discard comments
    {"-pch-stats", no_argument, OPT_PCH_STATS},  // FPC
    {"-pch-key", required_argument, OPT_PCH_KEY},
    {"-pch", required_argument, OPT_PCH},
    {"-no-prune", no_argument, OPT_NO_PRUNE},
    {"-help", no_argument, OPT_HELP},
    {"v", no_argument, OPT_VERSION},
    {"-version", no_argument, OPT_VERSION},
    {0},
  };
  int opt;
  while ((opt = optparse(argc, argv, options)) != -1) {
    switch (opt) {
    default: assert(false); break;
    case OPT_HELP:
      usage(stdout);
      return 0;
    case OPT_VERSION:
      show_version("cpp", -1);
      return 0;
    case 'I':
      add_inc_path(INC_NORMAL, optarg);
      pch_set_args(2, (char*[]){"-I", optarg});
      break;
    case OPT_ISYSTEM:
      add_inc_path(INC_SYSTEM, optarg);
      pch_set_args(2, (char*[]){"-isystem", optarg});
      break;
    case OPT_IDIRAFTER:
      add_inc_path(INC_AFTER, optarg);
      pch_set_args(2, (char*[]){"-idirafter", optarg});
      break;
    case 'D':
      define_macro(optarg);
      pch_set_args(2, (char*[]){"-D", optarg});
      break;
    case 'U':
      undef_macro(optarg, NULL);
      pch_set_args(2, (char*[]){"-U", optarg});
      break;
    case 'C':
      set_preserve_comment(true);
      pch_dir = "";  // FPC: no cache (comments in the output)
      break;
    case OPT_PCH:
      if (pch_dir == NULL)
        pch_dir = optarg;
      break;
    case OPT_PCH_STATS:
      pch_stats = 1;
      break;
    case OPT_NO_PRUNE:
      pch_prune = false;
      break;
    case OPT_PCH_KEY:  // The system headers' version: not checked one by one
      pch_set_args(2, (char*[]){"--pch-key", optarg});
      pch_trust_system = true;
      break;
    case '?':
      fprintf(stderr, "Warning: unknown option: %s\n", argv[optind - 1]);
      break;
    }
  }

  int iarg = optind;
  if (argc - iarg != 1 || (pch_dir != NULL && *pch_dir == '\0'))
    pch_dir = NULL;  // FPC: one file
  if (iarg < argc) {
    for (int i = iarg; i < argc; ++i) {
      const char *filename = argv[i];
      FILE *fp;
      if (!is_file(filename) || (fp = fopen(filename, "r")) == NULL)
        error("Cannot open file: %s\n", filename);
      preprocess_main(fp, filename);  // FPC
      fclose(fp);
    }
  } else {
    preprocess(stdin, "*stdin*");
  }
  if (ofp != stdout)
    fclose(ofp);
  return 0;
}
