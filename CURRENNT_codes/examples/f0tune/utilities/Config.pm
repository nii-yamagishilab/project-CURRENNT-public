#!/usr/bin/perl
# ----------------------------------------------------------------- #
#           The HMM-Based Speech Synthesis System (HTS)             #
#           developed by HTS Working Group                          #
#           http://hts.sp.nitech.ac.jp/                             #
# ----------------------------------------------------------------- #
#                                                                   #
#  Copyright (c) 2001-2012  Nagoya Institute of Technology          #
#                           Department of Computer Science          #
#                                                                   #
#                2001-2008  Tokyo Institute of Technology           #
#                           Interdisciplinary Graduate School of    #
#                           Science and Engineering                 #
#                                                                   #
#                2008       University of Edinburgh                 #
#                           Centre for Speech Technology Research   #
#                                                                   #
# All rights reserved.                                              #
#                                                                   #
# Redistribution and use in source and binary forms, with or        #
# without modification, are permitted provided that the following   #
# conditions are met:                                               #
#                                                                   #
# - Redistributions of source code must retain the above copyright  #
#   notice, this list of conditions and the following disclaimer.   #
# - Redistributions in binary form must reproduce the above         #
#   copyright notice, this list of conditions and the following     #
#   disclaimer in the documentation and/or other materials provided #
#   with the distribution.                                          #
# - Neither the name of the HTS working group nor the names of its  #
#   contributors may be used to endorse or promote products derived #
#   from this software without specific prior written permission.   #
#                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       #
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS #
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON #
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    #
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           #
# POSSIBILITY OF SUCH DAMAGE.                                       #
# ----------------------------------------------------------------- #


# Settings ==============================
$fclf  = 'HTS_TTS_ENG';
$fclv  = '1.0';
$dset  = 'BC2011';
@TSPKR = split(' ', 'nancy');
@ASPKR = split(' ', 'dummy');
$qnum  = '001';
$ver   = '1';

@SET        = ('cmp','dur');
@cmp        = ('mgc','lf0','bap');
@dur        = ('dur');
$ref{'cmp'} = \@cmp;
$ref{'dur'} = \@dur;

%nstate = ('cmp' => '5',     # number of states
           'dur' => '1');

%vflr = ('mgc' => '0.01',           # variance floors
         'lf0' => '0.01',
         'bap' => '0.01',
         'dur' => '0.01');

%thr  = ('mgc' => '000',            # minimum likelihood gain in clustering
         'lf0' => '000',
         'bap' => '000',
         'dur' => '000');

%mdlf = ('mgc' => '1.0',            # tree size control param. for MDL
         'lf0' => '1.0',
         'bap' => '1.0',
         'dur' => '1.0');

%mocc = ('mgc' => '10.0',           # minimum occupancy counts
         'lf0' => '10.0',
         'bap' => '10.0',
         'dur' => ' 5.0');

%gam  = ('mgc' => '000',            # stats load threshold
         'lf0' => '000',
         'bap' => '000',
         'dur' => '000');

%t2s  = ('mgc' => 'cmp',            # feature type to mmf conversion
         'lf0' => 'cmp',
         'bap' => 'cmp',
         'dur' => 'dur');

%strb = ('mgc' => '1',              # stream start
         'lf0' => '2',
         'bap' => '5',
         'dur' => '1');

%stre = ('mgc' => '1',              # stream end
         'lf0' => '4',
         'bap' => '5',
         'dur' => '5');

%msdi = ('mgc' => '0',              # msd information
         'lf0' => '1',
         'bap' => '0',
         'dur' => '0');

%strw = ('mgc' => '1.0',            # stream weights
         'lf0' => '1.0',
         'bap' => '0.0',
         'dur' => '1.0');

%ordr = ('mgc' => '60',     # feature order
         'lf0' => '1',
         'bap' => '25',
         'vuv' => '1',
         'lab' => '382',
         'dur' => '5');

%nwin = ('mgc' => '3',     # number of windows
         'lf0' => '3',
         'bap' => '3',
         'dur' => '0');

%gvthr  = ('mgc' => '000',          # minimum likelihood gain in clustering for GV
           'lf0' => '000',
           'bap' => '000');

%gvmdlf = ('mgc' => '1.0',          # tree size control for GV
           'lf0' => '1.0',
           'bap' => '1.0');

%gvgam  = ('mgc' => '000',          # stats load threshold for GV
           'lf0' => '000',
           'bap' => '000');

@slnt = split(' ', 'pau #');        # silent and pause phoneme

#%mdcp = ('dy' => 'd',              # model copy
#         'A'  => 'a',
#         'I'  => 'i',
#         'U'  => 'u',
#         'E'  => 'e',
#         'O'  => 'o');


# Speech Analysis/Synthesis Setting ==============
# speech analysis
$sr = 48000;   # sampling rate (Hz)
$fs = 240; # frame period (point)
$fl = 1200;   # frame length (point)
$fp = 4096;     # fft length (point)
$fw = 0.77;   # frequency warping
$gm = 0;      # pole/zero representation weight
$lg = 1;     # use log gain instead of linear gain
@F0RANGETMP = split(' ', 'nancy 40 800 dummy 40 800');
for ($i=0;$i <= $#F0RANGETMP; $i+=3) {
   $F0RANGE{$F0RANGETMP[$i]}{'lower'} = $F0RANGETMP[$i+1];
   $F0RANGE{$F0RANGETMP[$i]}{'upper'} = $F0RANGETMP[$i+2];
}
$accenttype = "GAM";
if ($accenttype eq "GAM") {
   $accopt = "-vg";
}
elsif ($accenttype eq "EDI") {
   $accopt = "-ve";
}
else {
   $accopt = "-vr";
}

# speech synthesis
$pf_mcp   = 0.3; # postfiltering factor for mel-cepstrum
$pf_lsp   = 0.7; # postfiltering factor for LSP
$il       = 576;        # length of impulse response
$co       = 2047;            # order of cepstrum to approximate mel-cepstrum
$wincoef  = 0.42;
$winpower = 19.1106090545654296875;


# Speaker adaptation Setting ============
$spkrPat = "\"*/BC2011_%%%%%_*\"";       # speaker name pattern

# regression classes
%dect = ('mgc' => '500.0',    # occupancy thresholds for regression classes (dec)
         'lf0' => '100.0',    # set thresholds in less than adpt and satt
         'bap' => '100.0',
         'dur' => '100.0');

$nClass  = 32;                            # number of regression classes (reg)

# transforms
%nblk = ('mgc' => '3',       # number of blocks for transforms
         'lf0' => '1',
         'bap' => '3',
         'dur' => '1');

%band = ('mgc' => '60',       # band width for transforms
         'lf0' => '1',
         'bap' => '25',
         'dur' => '0');

$bias{'cmp'} = 'TRUE';               # use bias term for MLLRMEAN/CMLLR
$bias{'dur'} = 'TRUE';
$tran        = 'feat';             # transformation kind (mean -> MLLRMEAN, cov -> MLLRCOV, or feat -> CMLLR)

# adaptation
%adpt = ('mgc' => '1000.0',       # occupancy thresholds for adaptation
         'lf0' => '200.0',
         'bap' => '500.0',
         'dur' => '500.0');

$tknd{'adp'}   = 'dec';            # tree kind (dec -> decision tree or reg -> regression tree (k-means))
$dcov          = 'FALSE';             # use diagonal covariance transform for MLLRMEAN
$usemaplr      = 'TRUE';            # use MAPLR adaptation for MLLRMEAN/CMLLR
$usevblr       = 'FALSE';             # use VBLR adaptation for MLLRMEAN
$sprior        = 'TRUE';  # use structural prior for MAPLR/VBLR with regression class tree
$priorscale    = 1.0;            # hyper-parameter for SMAPLR adaptation
$nAdapt        = 3;              # number of iterations to reestimate adaptation xforms
$addMAP        = 1;                # apply additional MAP estimation after MLLR adaptation
$maptau{'cmp'} = 50.0;             # hyper-parameters for MAP adaptation
$maptau{'dur'} = 50.0;

# speaker adaptive training
%satt = ('mgc' => '10000.0',    # occupancy thresholds for adaptive training
         'lf0' => '2000.0',
         'bap' => '5000.0',
         'dur' => '5000.0');

$tknd{'sat'} = 'dec';           # tree kind (dec -> decision tree or reg -> regression tree (k-means))
$nSAT        = 3;                  # number of SAT iterations


# Modeling/Generation Setting ==============
# modeling
$nState      = 5;        # number of states
$nIte        = 5;         # number of iterations for embedded training
$pr_nIte     = 200;       # number of iterations for phase reconstruction
$beam        = '1500 100 5000'; # initial, inc, and upper limit of beam width
$maxdev      = 10;        # max standard dev coef to control HSMM maximum duration
$mindur      = 5;        # min state duration to be evaluated
$wf          = 5000;        # mixture weight flooring
$initdurmean = 3.0;             # initial mean of state duration
$initdurvari = 10.0;            # initial variance of state duration
$daem        = 1;          # DAEM algorithm based parameter estimation
$daem_nIte   = 10;     # number of iterations of DAEM-based embedded training
$daem_alpha  = 1.0;     # schedule of updating temperature parameter for DAEM

# generation
$pgtype     = 0;     # parameter generation algorithm (0 -> Cholesky,  1 -> MixHidden,  2 -> StateHidden)
$maxEMiter  = 20;  # max EM iteration
$EMepsilon  = 0.0001;  # convergence factor for EM iteration
$useGV      = 1;      # turn on GV
$useSigPF   = 1;   # turn on signal processing based postfilter
$exSilNNPF  = 1;  # exclude silence for DNN-based postfilter
$maxGViter  = 50;  # max GV iteration
$GVepsilon  = 0.0001;  # convergence factor for GV iteration
$minEucNorm = 0.01; # minimum Euclid norm for GV iteration
$stepInit   = 1.0;   # initial step size
$stepInc    = 1.2;    # step size acceleration factor
$stepDec    = 0.5;    # step size deceleration factor
$hmmWeight  = 1.0;  # weight for HMM output prob.
$gvWeight   = 1.0;   # weight for GV output prob.
$optKind    = 'NEWTON';  # optimization method (STEEPEST, NEWTON, or LBFGS)
$nosilgv    = 1;    # GV without silent and pause phoneme
$cdgv       = 1;       # context-dependent GV

# Parallel ===============
$nPara      = 10;
$adpnPara   = 10;
$gennPara   = 10;
$USEPBS     = 0;


# Directories & Commands ===============
# project directories
$prjdir        = '/home/smg/wang/PROJ/DL/DNNAM';
$datdir        = "$prjdir/data";
$hmmdir        = "$prjdir/hmm";
$hmmgendir     = "$hmmdir/gen/qst${qnum}/ver${ver}/SI/$pgtype";
$nndatdir      = "$prjdir/nndata";
$dnndir        = "$prjdir/dnn";
$dnngendir     = "$dnndir/gen";
$aedatdir      = "$prjdir/aedata";
$aednndir      = "$prjdir/aednn";
$iaednndir     = "$prjdir/iaednn";
$pfdatdir      = "$prjdir/pfdata";
$iaednnpfdir   = "$prjdir/iaednnpf";

# Perl
$PERL = '/usr/bin/perl';

# wc
$WC = '/usr/bin/wc';

# HTS commands
$HCOMPV    = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HCompV';
$HLIST     = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HList';
$HINIT     = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HInit';
$HREST     = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HRest';
$HEREST    = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HERest';
$HHED      = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HHEd';
$HSMMALIGN = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HSMMAlign';
$HMGENS    = '/home/smg/wang/TOOL/bin/HTS-2.3alpha-NN/bin/HMGenS';
$ENGINE    = '/home/smg/takaki/SRC/hts_engine_API-1.09/bin/hts_engine';

# Neural Network Data commands
$LF0IP = '/home/smg/wang/TOOL/NeuralNetworkData/bin/F0Interpolation';
$F0VUV = '/home/smg/wang/TOOL/NeuralNetworkData/bin/F0VUVComposition';

# SPTK commands
$X2X      = '/home/smg/wang/TOOL/bin/sptk/bin/x2x';
$SOPR     = '/home/smg/wang/TOOL/bin/sptk/bin/sopr';
$LSP2LPC  = '/home/smg/wang/TOOL/bin/sptk/bin/lsp2lpc';
$MGC2MGC  = '/home/smg/wang/TOOL/bin/sptk/bin/mgc2mgc';
$MERGE    = '/home/smg/wang/TOOL/bin/sptk/bin/merge';
$LSPCHECK = '/home/smg/wang/TOOL/bin/sptk/bin/lspcheck';
$MGC2SP   = '/home/smg/wang/TOOL/bin/sptk/bin/mgc2sp';
$BCUT     = '/home/smg/wang/TOOL/bin/sptk/bin/bcut';
$VSTAT    = '/home/smg/wang/TOOL/bin/sptk/bin/vstat';
$NAN      = '/home/smg/wang/TOOL/bin/sptk/bin/nan';
$MINMAX   = '/home/smg/wang/TOOL/bin/sptk/bin/minmax';
$MGCEP    = '/home/smg/wang/TOOL/bin/sptk/bin/mcep';
$LPC2LSP  = '/home/smg/wang/TOOL/bin/sptk/bin/lpc2lsp';
$PITCH    = '/home/smg/wang/TOOL/bin/sptk/bin/pitch';
$MLPG     = '/home/smg/wang/TOOL/bin/sptk/bin/mlpg';
$MCEP     = '/home/smg/wang/TOOL/bin/sptk/bin/mcep';

# postfilter
$MCPF  = '/home/smg/wang/TOOL/bin/postfilter/bin/mcpf';
$LSPPF = '/home/smg/wang/TOOL/bin/postfilter/bin/lsppf';

# STRAIGHT
$STRAIGHT_MCEP  = '/home/smg/wang/TOOL/bin/straight/bin/straight_mcep';
$TEMPO          = '/home/smg/wang/TOOL/bin/straight/bin/tempo';
$STRAIGHT_BNDAP = '/home/smg/wang/TOOL/bin/straight/bin/straight_bndap';
$SYNTHESIS_FFT  = '/home/smg/wang/TOOL/bin/straight/bin/synthesis_fft';

# bndap2ap
$BNDAP2AP = '/home/smg/takaki/SRC/BandAperiodicity/bin/bndap2ap';

# flite_hts_text_processing
$FLITE_HTS_TEXT_PROCESSING = '/home/smg/takaki/SRC/flite+hts_engine-1.01-VCTK/bin/flite_hts_text_processing';

# Switch for DATA ============================
$DAT_MKEMV = 1; # preparing data environments
$DAT_ANASP = 1; # speech analysis
$DAT_CMPFT = 1; # composing features
$DAT_ANATT = 1; # text analysis
$DAT_CPALB = 1; # copying generating labels for adapt speaker
$DAT_MLSCP = 1; # making master label files and scp

# Switch for training HMM ================================
$HMM_MKEMV = 1; # preparing environments
$HMM_HCMPV = 1; # computing a global variance
$HMM_IN_RE = 1; # initialization & reestimation
$HMM_MMMMF = 1; # making a monophone mmf
$HMM_ERST0 = 1; # embedded reestimation (monophone)
$HMM_MN2FL = 1; # copying monophone mmf to fullcontext one
$HMM_ERST1 = 1; # embedded reestimation (fullcontext)
$HMM_CXCL1 = 1; # tree-based context clustering
$HMM_ERST2 = 1; # embedded reestimation (clustered)
$HMM_UNTIE = 1; # untying the parameter sharing structure
$HMM_ERST3 = 1; # embedded reestimation (untied)
$HMM_CXCL2 = 1; # tree-based context clustering
$HMM_ERST4 = 1; # embedded reestimation (re-clustered)
$HMM_FALGN = 1; # forced alignment for no-silent GV
$HMM_MCDGV = 1; # making global variance
$HMM_MKUNG = 1; # making unseen models (GV)
$HMM_MKUN1 = 1; # making unseen models (speaker independent)
$HMM_PGEN1 = 1; # generating speech parameter sequences (speaker independent)
$HMM_WGEN1 = 1; # synthesizing waveforms (speaker independent)
$HMM_SANA1 = 1; # structure analysis (SI)
$HMM_REGTR = 0; # building regression-class trees for adaptation
$HMM_ADPT1 = 0; # speaker adaptation (speaker independent)
$HMM_PGEN2 = 0; # generating speech parameter sequences (speaker adapted)
$HMM_WGEN2 = 0; # synthesizing waveforms (speaker adapted)
$HMM_SPKAT = 0; # speaker adaptive training (SAT)
$HMM_MKUN2 = 0; # making unseen models (SAT)
$HMM_PGEN3 = 0; # generating speech parameter sequences (SAT)
$HMM_WGEN3 = 0; # synthesizing waveforms (SAT)
$HMM_ADPT2 = 0; # speaker adaptation (SAT)
$HMM_PGEN4 = 0; # generate speech parameter sequences (SAT+adaptation)
$HMM_WGEN4 = 0; # synthesizing waveforms (SAT+adaptation)
$HMM_CONVM = 0; # converting mmfs to the hts_engine file format
$HMM_ENGIN = 0; # synthesizing waveforms using hts_engine

# Switch for neural network Data ============================
$DTN_MKEMV = 0; # preparing data environments
$DTN_FALGN = 0; # forced alignment
$DTN_EXLAB = 0; # extracting label features
$DTN_WP_SP = 0; # making warped sp
$DTN_EXSSP = 0; # making sp excluding silence
$DTN_DLMGC = 0; # calculating mgc delta
$DTN_DLLF0 = 1; # making voice/unvoice value and calculating lf0 delta
$DTN_DLBAP = 0; # calculating bap delta
$DTN_CLVAR = 0; # calculating global variance of output features
$DTN_MKSCP = 0; # making scp for neural network

# Switch for training DNN ============================
$DNN_MKEMV = 1; # preparing environments for neural network
$DNN_TNDNN = 1; # training DNN and synthesize features
$DNN_GNWAV = 1; # generating wav

# Switch for auto-encoder DATA ============================
$AED_MKEMV = 1; # preparing environments
$AED_MKAEF = 1; # making auto-encoder features
$AED_MKSCP = 1; # making scp

# Switch for training auto-encoder DNN ============================
$AEN_MKEMV = 1; # preparing environments
$AEN_TNDNN = 1; # training DNN and synthesize features
$AEN_GNWAV = 1; # generating wav

# Switch for training integrated AEDNN ============================
$IAN_MKEMV = 1; # preparing environments
$IAN_TRNNN = 1; # training integrated AEDNN
$IAN_GNWAV = 1; # generating wav

# Switch for postfilter Data ============================
$PFD_MKEMV = 1; # preparing environments
$PFD_IGNDT = 1; # generating training and test data
$PFD_IEXSL = 1; # excluding silence part from training data
$PFD_IMSCP = 1; # making scp

# Switch for training postfilter for IAEDNN ============================
$IPF_MKEMV = 1; # preparing environments
$IPF_TRNPF = 1; # training DNN postfilter
$IPF_GNWAV = 1; # generating wav

1;
