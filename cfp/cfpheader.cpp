#include "zfp/array1.hpp"
#include "zfp/array2.hpp"
#include "zfp/array3.hpp"
#include "zfp/array4.hpp"
#include "zfp/internal/codec/zfpheader.hpp"
#include "zfp/internal/cfp/header.h"
#include "zfp/internal/cfp/array1f.h"
#include "zfp/internal/cfp/array1d.h"
#include "zfp/internal/cfp/array2f.h"
#include "zfp/internal/cfp/array2d.h"
#include "zfp/internal/cfp/array3f.h"
#include "zfp/internal/cfp/array3d.h"
#include "zfp/internal/cfp/array4f.h"
#include "zfp/internal/cfp/array4d.h"

#include "template/template.h"

#define CFP_HEADER_TYPE cfp_header
#define ZFP_HEADER_TYPE zfp::array::header

#include "template/cfpheader.cpp"
