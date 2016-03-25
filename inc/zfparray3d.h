#ifndef ZFP_ARRAY3D_H
#define ZFP_ARRAY3D_H

#include "zfpcodec3d.h"
#include "zfparray3.h"

namespace ZFP {
  typedef Array3< double, Codec3d<MemoryBitStream> > Array3d;
}

#endif
