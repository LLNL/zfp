#ifndef ZFP_ARRAY1F_H
#define ZFP_ARRAY1F_H

#include "zfpcodec1f.h"
#include "zfparray1.h"

namespace ZFP {
  typedef Array1< float, Codec1f<MemoryBitStream> > Array1f;
}

#endif
