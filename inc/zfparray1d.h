#ifndef ZFP_ARRAY1D_H
#define ZFP_ARRAY1D_H

#include "zfpcodec1d.h"
#include "zfparray1.h"

namespace ZFP {
  typedef Array1< double, Codec1d<MemoryBitStream> > Array1d;
}

#endif
