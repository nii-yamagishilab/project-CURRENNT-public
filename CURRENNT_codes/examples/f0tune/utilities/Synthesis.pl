#!/usr/bin/perl
###########################################################################
## ---------------------------------------------------------------------  #
##  Neural Network Training Tool                                          #
## ---------------------------------------------------------------------  #
##                                                                        #
##  Copyright (c) 2014  National Institute of Informatics                 #
##                                                                        #
##  THE NATIONAL INSTITUTE OF INFORMATICS AND THE CONTRIBUTORS TO THIS    #
##  WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING  #
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT    #
##  SHALL THE NATIONAL INSTITUTE OF INFORMATICS NOR THE CONTRIBUTORS      #
##  BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY   #
##  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,       #
##  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS        #
##  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE   #
##  OF THIS SOFTWARE.                                                     #
###########################################################################
##                         Author: Shinji Takaki                          #
##                         Date:   14 November 2014                       #
##                         Contact: takaki@nii.ac.jp                      #
###########################################################################

$| = 1;

if ( @ARGV < 1 ) {
   print "usage: DNNTraining.pl Config.pm Utils.pm\n";
   exit(0);
}

# load configuration variables
require( $ARGV[0] );
require( $ARGV[1] );

$jobflag = "$ARGV[2]";
$jobopt1 = "$ARGV[3]";
$jobopt2 = "$ARGV[4]";
$jobopt3 = "$ARGV[5]";
$jobopt4 = "$ARGV[6]";
$jobopt5 = "$ARGV[7]";
$jobopt6 = "$ARGV[8]";
$gennPara = "$ARGV[9]";
$synwav  = "$ARGV[10]";
$onlywav = "$ARGV[11]";
$prjdir        = "$jobopt5";
$nndatdir      = "$jobopt6";
$dnngendir     = "$jobopt5";

# set switch
$DNN_MKEMV = 0;    # preparing environments for neural network
$DNN_TNDNN = 0;    # training DNN and synthesize features
$DNN_GNWAV = 0;    # generating wav

$DNN_MKEMV = 1 if ( $jobflag eq "DNN_MKEMV" );
$DNN_TNDNN = 1 if ( $jobflag eq "DNN_TNDNN" );
$DNN_GNWAV = 1 if ( $jobflag eq "DNN_GNWAV" );

# data location file
$scp{'gen'} = "$jobopt4";

# generating wav
if ($DNN_GNWAV) {
   print_time("generating wav");
	
   $p = $jobopt1;
   make_parallel_scp( $scp{'gen'}, $gennPara, $p );
   
   $useMLPG = $jobopt2;
   
   $dir = "$dnngendir";
   
   if ($onlywav){
       gen_wave_only( $dir, $dir, $dir, $dir, "$scp{'gen'}.$p", ".htk", 1, $useMLPG, 1);
   }else{
       gen_wave( $dir, $dir, $dir, $dir, "$scp{'gen'}.$p", ".htk", 1, $useMLPG, $synwav);
   }

}

print_time("Done");

##################################################################################################
