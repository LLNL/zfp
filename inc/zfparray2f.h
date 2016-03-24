#ifndef ZFP_ARRAY2F_H
#define ZFP_ARRAY2F_H

#include "zfpcodec2f.h"
#include "zfparray2.h"

namespace ZFP {
  typedef Array2< float, Codec2f<MemoryBitStream> > Array2f;
}

#endif
