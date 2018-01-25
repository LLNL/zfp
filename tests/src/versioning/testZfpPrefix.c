#include "zfp.h"

/* any compiler error/warning -> test failure */

int main()
{
  (ZFP_V5_CODEC);

  zfp_v5_stream* stream = zfp_v5_stream_open(NULL);
  zfp_v5_stream_close(stream);

  return 0;
}
