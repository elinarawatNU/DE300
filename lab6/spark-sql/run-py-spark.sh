# export PYSPARK_DRIVER_PYTHON=python3 only in cluster mode
/bin/rm -r -f data

export PYSPARK_PYTHON=../demos/bin/python3
/opt/spark/bin/spark-submit --archives ../demos.tar.gz#demos process.py

