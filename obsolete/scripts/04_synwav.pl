#!/usr/bin/perl
###############################################################
## General Description:
##		This script generates the output acoustic features and 
##		synthesis the speech waveform
## Usage:
##		$nndatadir: this folder is copied from the DNNAM project
##					it contains the var and win
##					var (variance of the acoustic feature)
##						\- mgc.var ...
##						\-
##					win (window for computing delta coefficients)
##						\- mgc.win ...
##		
## Notes:
##		1. Please configure data_config.py before running
##		2. Please generate the output before using this script
##		3. Please make sure data.nc contains the mean and variance
##			for output data
###############################################################

use Cwd;

# path
$prjdir    = $ARGV[0];
$input     = $ARGV[1];#"$prjdir/data.nc1";
$network   = $ARGV[2];#"trained_network.jsn";
$moddir    = $ARGV[3];
$outputdir = $ARGV[4];
$outputdir2= $ARGV[5];#
$nndatadir = $ARGV[6];
$spgdir    = $ARGV[8];
$lf0dir    = $ARGV[9];
$bapdir    = $ARGV[10];
$pgflag    = $ARGV[11];

$fgendata   = $ARGV[12];
$fsplitdata = $ARGV[13];
$fsynwave   = $ARGV[14];

$mvfile=$ARGV[15];
$we1  = $ARGV[16];
$we2  = $ARGV[17];
$we3  = $ARGV[18];

$tapl = $ARGV[19];
$tapg = $ARGV[20];
$mdnp = $ARGV[21];

require($ARGV[7]) || die "Can't find $ARGV[7]";
# binary executable file
#$currennt  = "/home/smg/wang/TOOL/CURRENNT_219/build/currennt";

# Process
if ($GENDATA && $fgendata){
    print_time("--- 4.1 generating from NN network ---");
    if (-e "$moddir/config_syn.cfg"){
    }else{
	SelfSystem("cp ./utilities/config_syn.cfg $moddir/config_syn.cfg");
    }
    # generate the data
    $curdir = getcwd;
    chdir($moddir);
    print "Entering $moddir\n";
    print "\n-------------------- CURRENNNT generating log ----------------\n";
    SelfSystem("cat ./config_syn.cfg");
    $command = "$currennt --options_file config_syn.cfg";
    $command = "$command --ff_input_file $input";
    $command = "$command --ff_output_file $outputdir --network $network";
    if (-e $outputdir){
    }else{
	mkdir "$outputdir";
    }
    if ($we1 ne "NONE"){
	$command = "$command --weBank $we1 --weIDDim $we2 --weDim $we3";
	$command = "$command --weExternal 1";
    }
    
    if ($tapl ne "NONE"){
	$command = "$command --output_from $tapl --output_from_gate $tapg";
    }
    
    if ($mdnp ne "NONE"){
	$command = "$command --mdn_samplePara $mdnp";
    }
    print "\n$command\n";
    SelfSystem($command);
    print "--------------------------------------------------------------\n";
    chdir($curdir);
    print "Entering $curdir\n";
}else{
    print_time("--- 4.1 skip generating from NN network ---");
}

if ($SPLITDATA && $fsplitdata){
    print_time("--- 4.2 formmating output of NN to target data  ---");
    mkdir $outputdir;
    if (-e "$prjdir/data_config.py"){
	
    }else{
	die "Can't find $prjdir/data_config.py";
    }
    $command = "python ./utilities/GenSynData.py";
    $command = "$command $prjdir/data_config.py $outputdir";
    $command = "$command $mlpgFlag $outputdir2 $mvfile";
    SelfSystem($command);
}else{
    print_time("--- 4.2 skip formmating output of NN  ---");
}

if ($SYNWAVE && $fsynwave){
    print_time("--- 4.3 generating speech waveform  ---");
    # synthesis
    $batch = 1;
    while($batch <= $gennPara){
	$command = "perl ./utilities/Synthesis.pl ./utilities/Config.pm";
	$command = "$command ./utilities/Utils.pm DNN_GNWAV $batch";
	$command = "$command $mlpgFlag 0 $outputdir/gen.scp $outputdir";
	$command = "$command $nndatadir $gennPara $genWav $onlywav";
	$command = "$command $spgdir $lf0dir $bapdir $pgflag";
	print "$command\n";
	unless(-e "$outputdir/gen.scp"){ 
	    die "Is gen.scp not generated?\n Or you may copied it from somewhere else";
	}
	if ($DEBUG){
	    print "$command\n";
	}else{
	    defined ($jobID = fork) or die "Cannot fork: $!";
	    unless($jobID) {
		exec($command);
		die "cannot exec: $!";
	    }
	    $nJobs[$batch-1] = $jobID;
        }
	$batch = $batch + 1;
    }
    foreach $job (@nJobs){
	waitpid($job, 0);
    }
}else{
    print_time("--- 4.3 skip generating speech waveform  ---");
}
