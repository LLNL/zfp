#ifndef ZFP_ARRAY2D_H
#define ZFP_ARRAY2D_H

#include "zfpcodec2d.h"
#include "zfparray2.h"

namespace ZFP {
  typedef Array2< double, Codec2d<MemoryBitStream> > Array2d;
}

#endif
