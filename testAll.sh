while getopts 'v' flag; do
  case "${flag}" in
    v) ctest_verbose='-V' ;;
  esac
done

sh debug.sh
cd build-debug
ctest ${ctest_verbose}
cd ..

sh release.sh
cd build-release
ctest ${ctest_verbose}
cd ..
