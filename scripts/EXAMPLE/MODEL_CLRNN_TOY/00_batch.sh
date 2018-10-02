#!/bin/sh

# path to CURRENNT
currennt=currennt

# path to data.nc
data=../DATA_ZEROINPUT/data.nc1

# path to data.mv (used to de-normalize the generated data)
mv=../DATA_ZEROINPUT/data.mv

# RNN
${currennt} --options_file config.cfg --network network.jsn_RNN --train_file ${data}
mv trained_network.jsn ./NETWORK/trained_RNN.jsn
rm ./OUTPUT/OUTPUT_RNN/*
${currennt} --options_file config_syn.cfg --ff_input_file ${data} --ff_output_file ./OUTPUT/OUTPUT_RNN --network ./NETWORK/trained_RNN.jsn --datamv ${mv}

# CLRNN
${currennt} --options_file config.cfg --network network.jsn_CLRNN --train_file ${data}
mv trained_network.jsn ./NETWORK/trained_CLRNN.jsn
rm ./OUTPUT/OUTPUT_CLRNN/*
${currennt} --options_file config_syn.cfg --ff_input_file ${data} --ff_output_file ./OUTPUT/OUTPUT_CLRNN --network ./NETWORK/trained_CLRNN.jsn --datamv ${mv}

# DBLSTM
${currennt} --options_file config.cfg --network network.jsn_DBLSTM --train_file ${data} --lstm_forget_gate_bias 5.0
mv trained_network.jsn ./NETWORK/trained_DBLSTM.jsn
rm ./OUTPUT/OUTPUT_DBLSTM/*
${currennt} --options_file config_syn.cfg --ff_input_file ${data} --ff_output_file ./OUTPUT/OUTPUT_DBLSTM --network ./NETWORK/trained_DBLSTM.jsn --datamv ${mv}

# CLDBLSTM
${currennt} --options_file config.cfg --network network.jsn_CLDBLSTM --train_file ${data} --lstm_forget_gate_bias 5.0
mv trained_network.jsn ./NETWORK/trained_CLDBLSTM.jsn
rm ./OUTPUT/OUTPUT_CLDBLSTM/*
${currennt} --options_file config_syn.cfg --ff_input_file ${data} --ff_output_file ./OUTPUT/OUTPUT_CLDBLSTM --network ./NETWORK/trained_CLDBLSTM.jsn --datamv ${mv}


echo "The output of NN is put in ./OUTPUT/*. The output is a htk-format file, float 32bit, BIG-endian\n"
echo "Please use the ioTools in pyTools to read the htk file\n"
echo ">> from ioTools import readwrite as py_rw\n"
echo ">> data1 = py_rw.read_htk(PATH_TO_FILE, 'f4', 'b')\n"
echo "To read the target output data"
echo ">> data2 = py_rw.read_raw_mat('../DATA_ZEROINPUT/TEMPDATA/test_out.bin', 1)\n"
echo "Then, plot and compare the generated data agains the target data"
