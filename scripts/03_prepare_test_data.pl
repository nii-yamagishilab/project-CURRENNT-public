#!/usr/bin/perl
###############################################################
## General Description:
## Notes:
###############################################################

use Cwd;
use File::Basename;

# data dir
$prjdir    = $ARGV[0]; 
# mean and variance 
$datamv    = $ARGV[1]; 
# buff dir
$buffdir   = $ARGV[2];

require($ARGV[3]) || die "Can't not load $ARGV[3]";

## reserved options
$flagMultiNC = 1;		

mkdir $buffdir;


if ($TEST_FLAG_SCP){
    print_time("--- 3.1 generating scps for test data ---");

    # Delete the old *.info if new .scp is utilized
    SelfSystem("rm $prjdir/*.info");
    if(-e "$prjdir/data_config.py"){
	
    }else{
	print "Can't find $prjdir/data_config.py";
	die "Please use ./template/data_config.py_test as template\n";
    }
    
    # prepare to list for packaging data
    SelfSystem("python ./utilities/PrePareData.py $prjdir/data_config.py $prjdir");
    
    
    if (-e "$prjdir/all.scp"){
	
    }else{
	die "Fail to generate $prjdir/all.scp\n";
    }

    $scpFile = "$prjdir/all.scp";
    $dataScp = "$prjdir/gen.scp";
    open(IN_STR, "$scpFile");
    open(OUT_STR,">$dataScp");
    $count=1;
    while(<IN_STR>){
	chomp();
	$file = $_;
	open(IN_STR_2, "$file");
	while(<IN_STR_2>){
	    chomp();
	    @lineContents = split(' ', $_);
	    print OUT_STR "$lineContents[5]\n";
	}
	close(IN_STR_2);
    }
    close(OUT_STR);
    close(IN_STR);
    print "---\n";
    print "$prjdir/all.scp\n";
    print "$prjdir/gen.scp\n";
    
}else{
    print_time("--- 3.1 skip generating scps for test data ---");
}


if ($TEST_PREP_DAT){
    print_time("--- 3.2 package data for test set ---");
    
    if (-e "$prjdir/mask.txt"){
	print "Using the mask file $prjdir/mask.txt\n";
	$maskfile = "$prjdir/mask.txt";
    }else{
	$maskfile = "None";
    }
    if ( -e "$prjdir/normMask" ) {
        print "Using the normMask $prjdir/normMask\n";
        $normMaskBin = "$prjdir/normMask";
    }
    else {
        $normMaskBin = "None";
        print "Not using norm mask";
    }
    if ( -e "$prjdir/normMethod") {
	print "Using the normMethod $prjdir/normMethod\n";
	$normMethod = "$prjdir/normMethod";
    }else{
	$normMethod = "None";
	print "Not using norm method\n";
    }
    if (-e "$prjdir/all.scp"){
	
    }else{
	die "Can't find all.scp";
    }
    
    if ($dataPack ne ""){
	$command = "python ./utilities/PackData.py $prjdir/all.scp $datamv";
	# 1 0 1: reading/loading data, not calculate mean/std, normalize data
	# 1: add mean/std to each data.nc
	$command = "$command $buffdir 1 0 1 $maskfile 1 $normMaskBin $normMethod";
	SelfSystem($command);

    }else{
	print "Please configure dataPack in config.pm\n";
    }
}else{
    print_time("--- 3.2 skip packaging data for test set ---");
}

