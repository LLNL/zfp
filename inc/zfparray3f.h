#ifndef ZFP_ARRAY3F_H
#define ZFP_ARRAY3F_H

#include "zfpcodec3f.h"
#include "zfparray3.h"

namespace ZFP {
  typedef Array3< float, Codec3f<MemoryBitStream> > Array3f;
}

#endif
