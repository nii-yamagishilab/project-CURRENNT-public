#!/usr/bin/perl
use Cwd;
use File::Basename;
###############################################################
## General Description:
##		This script prepares the input data to CURRENT
## Notes:
##		Please configure data_config.py before running
###############################################################

$datadir = $ARGV[0];   # data directory (where data_config.py locates)
$buffadd = $ARGV[1];   # data buffer directory (where data.nc will locate)
require($ARGV[2]) || die "Can't find config.pm";

mkdir $buffadd, 0755;

if ( -e "$datadir/data_config.py" ) {
    print "Using $datadir/data_config.py\n";
}else{
    print "Found no $datadir/data_config.py\n";
    die "Please use ./template/data_config.py_train as template\n";
}

if ($FLAG_SCP) {
    
    if (-e "$datadir/data_config.py"){
	
    }else{
	die "Can't find $datadir/data_config.py";
    }
    
    # Delete the old *.info if new .scp is utilized
    SelfSystem("rm -f $datadir/*.info");
    print_time("--- 1.1 generating scps ---");

    # Prepare the data list
    $command = "python ./utilities/PrePareData.py $datadir/data_config.py $datadir";
    SelfSystem($command);
    
}else{
    print_time("--- 1.1 skip generating scps ---");
}

if ($PREP_DAT) {
    
    print_time("--- 1.2 package data ---");
    print("======     configuration info  ======\n");
    if ( -e "$datadir/all.scp" ){
	
    }else{
	die "Can't find $datadir/all.scp";
    }
    
    if ( -e "$datadir/mask.txt" ) {
        print "Using the mask file $datadir/mask.txt\n";
        $maskfile = "$datadir/mask.txt";
    }else {
        $maskfile = "None";
        print "Not using data mask\n";
    }
    
    if ( -e "$datadir/normMask" ) {
        print "Using the normMask $datadir/normMask\n";
        $normMaskBin = "$datadir/normMask";
    } else {
        $normMaskBin = "None";
        print "Not using norm mask\n";
    }

    if ( -e "$datadir/normMethod") {
	print "Using the normMethod $datadir/normMethod\n";
	$normMethod = "$datadir/normMethod";
    }else{
	$normMethod = "None";
	print "Not using norm method\n";
    }
    
    print "\n\n";
    
    if ($dataPack ne ""){
	
	$command = "python ./utilities/PackData.py $datadir/all.scp $datadir/data.mv";
	# 1 1 1: reading/loading data, calculate mean/std, normalize data
	# 0: don't add mean/std to each data.nc
	$command = "$command $buffadd 1 1 1 $maskfile 0 $normMaskBin $normMethod"; 	
	SelfSystem($command);
	
    }else{
	print "Please configure dataPack in config.pm\n";
    }
}else{
    print_time("--- 1.2 skip packaging data ---");
}

