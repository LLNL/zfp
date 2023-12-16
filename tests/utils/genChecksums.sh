#!/usr/bin/env bash
set -ex

MAX_DIM=4
SCALAR_TYPES=( "Float" "Double" "Int32" "Int64" )

mkdir -p checksumGenBuild
cd checksumGenBuild

cmake ../../.. -DZFP_WITH_OPENMP=OFF -DPRINT_CHECKSUMS=1
cmake --build . -- -j

for DIM in $(seq 1 $MAX_DIM);
do
  DIM_STR="${DIM}d"

  for SCALAR_STR in "${SCALAR_TYPES[@]}"
  do

    TEST_OUTPUT_FILE="test_output"
    TEMP_FILE="temp"
    TEMP_CHECKSUMS_FILE="temp_checksums"
    OUTPUT_FILE="${DIM_STR}${SCALAR_STR}.h"

    ctest -V -R "testZfpEncodeBlock${DIM_STR}${SCALAR_STR}" -O $TEMP_FILE
    cat "$TEMP_FILE" >> "$TEST_OUTPUT_FILE"

    ctest -V -R "testZfpEncodeBlockStrided${DIM_STR}${SCALAR_STR}" -O $TEMP_FILE
    cat "$TEMP_FILE" >> "$TEST_OUTPUT_FILE"

    ctest -V -R "testZfpDecodeBlock${DIM_STR}${SCALAR_STR}" -O $TEMP_FILE
    cat "$TEMP_FILE" >> "$TEST_OUTPUT_FILE"

    ctest -V -R "testZfpDecodeBlockStrided${DIM_STR}${SCALAR_STR}" -O $TEMP_FILE
    cat "$TEMP_FILE" >> "$TEST_OUTPUT_FILE"

    ctest -V -R "testZfpSerial${DIM_STR}${SCALAR_STR}" -O $TEMP_FILE
    cat "$TEMP_FILE" >> "$TEST_OUTPUT_FILE"

    grep -o '{UINT64C(0x.*' $TEST_OUTPUT_FILE > $TEMP_CHECKSUMS_FILE
    NUM_CHECKSUMS=$(wc -l < "$TEMP_CHECKSUMS_FILE")

    # create valid .h file

    echo "static const checksum_tuples _${DIM_STR}${SCALAR_STR}Checksums[${NUM_CHECKSUMS}] = {" > $OUTPUT_FILE
    cat $TEMP_CHECKSUMS_FILE >> $OUTPUT_FILE
    echo "};" >> $OUTPUT_FILE

    rm $TEST_OUTPUT_FILE $TEMP_FILE $TEMP_CHECKSUMS_FILE
  done
done
