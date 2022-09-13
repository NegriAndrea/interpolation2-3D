#!/bin/bash
#python setup.py build_ext --inplace

# production
#f2py -c  --f77flags='-Ofast -march=native -mtune=native' --f90flags='-Ofast -march=native -mtune=native' -m loopf loopF.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
#f2py -c --f77flags='-Ofast -march=native -mtune=native' --f90flags='-Ofast -march=native -mtune=native' -m loopffirstIndex loopFfirstIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
#f2py -c --f77flags='-Ofast -march=native -mtune=native' --f90flags='-Ofast -march=native -mtune=native' -m loopfsecondIndex loopFsecondIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1

#f2py -c --opt='-Ofast -march=native -mtune=native' -m loopf loopF.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
#f2py -c --opt='-Ofast -march=native -mtune=native' -m loopffirstIndex loopFfirstIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
#f2py -c --opt='-Ofast -march=native -mtune=native' -m loopfsecondIndex loopFsecondIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1

f2py -c -m loopf loopF.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
f2py -c -m loopffirstIndex loopFfirstIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
f2py -c -m loopfsecondIndex loopFsecondIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
gcc  -Ofast  -mtune=native -Wall -Wextra -pedantic -Warray-bounds  -fPIC -shared -o loopC.so   loopC.c
#gcc  -O3 -mtune=native -march=native -Wall -Wextra -pedantic -Warray-bounds  -fPIC -shared -o loopC.so loopC.c

#debug
#f2py --f90flags='-fcheck=all' -c -m loopf loopF.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1 -DF2PY_REPORT_ATEXIT
#f2py --f90flags='-fcheck=all' -c -m loopffirstIndex loopFfirstIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1 -DF2PY_REPORT_ATEXIT
#f2py --f90flags='-fcheck=all' -c -m loopfsecondIndex loopFsecondIndex.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1 -DF2PY_REPORT_ATEXIT
