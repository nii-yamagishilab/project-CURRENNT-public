#!/usr/bin/perl
###############################################################
## General Description:
##	        This script calculate the RMSE
## Usage:
##              
## Notes:
###############################################################

use Cwd;
use File::Basename;

#$projDir       = "/home/smg/wang/PROJ/WE/DNNAM/MODEL/nancy";
@sourceDirList = ("$ARGV[0]"); 
@nameDir       = ("$ARGV[1]");
@numDir        = ("$ARGV[2]");
require("$ARGV[3]") || die "Find no config.pm";
#$targetDir = "/home/smg/wang/DATA/speech/nancy/data";                              
#$fullabDir = "/home/smg/wang/DATA/speech/nancy/nndata/labels/full_align/test";
#$targetDir  = "/home/smg/wang/DATA/speech/nancy/nndata";
#$fullabDir  = "/home/smg/takaki/FEAT/nancy/nndata/full_align";

## reserved options
$pilotType = ".htk";
my @subDir = @RMSEtargetDir;   #("validationset", "validationset");
my @dataType  = @RMSEdataType; #(".lf0", ".mgc");
my @dataDim   = @RMSEdataDim;  #(1, 60);

my $filter    = $RMSEfilte;    #'^BC2011_';
my $resolut   = $LabResolut;   #50000;
my $escape    = $LabEscape;    #'\\\\-#\\\\+';


## 
mkdir "./tmp";
$i = 0;
foreach $sourceDir (@sourceDirList){
    opendir(D, $sourceDir) || die "Not found $sourceDir. Igore";
    @fileList = readdir(D);
    @fileList = grep {! /^\./} @fileList;
    @fileList = grep {/$filter/} @fileList;
    
    $tmpName   = basename($nameDir[$i]);
    $epoch     = $numDir[$i];
    $tempScp   = "./tmp/temp$tmpName$epoch.scp";
    $ctr = 0;
    print "$sourceDir\t";
    
    foreach $fileType (@dataType){
	#@tempFileList = @fileList;
    
	@tempFileList = grep {/$fileType$/} @fileList;
	open(OUT_STR, ">$tempScp$ctr");
    
	foreach $file (@tempFileList){
	    $baseName = $file;
	    $baseName =~ s/$fileType$/.lab/g;
	    print OUT_STR "$sourceDir/$file ";
	    unless(-e "$sourceDir/$file") {die "no $sourceDir/$file";}
	    print OUT_STR "$RMSEtargetDir[$ctr]/$file ";
	    unless(-e "$RMSEtargetDir[$ctr]/$file") {die "no $RMSEtargetDir/$file";}
	    print OUT_STR "$RMSEfull_labDir/$baseName\n";
	    unless(-e "$RMSEfull_labDir/$baseName") {die "no $RMSEfull_labDir/$baseName";}
	}    
	close(OUT_STR);
	$output = "$sourceDir/"."log"."$dataType[$ctr]"."RMSE";
	$command= "python ./utilities/RMSECal.py $tempScp$ctr";
	$command= "$command $dataDim[$ctr] $resolut $escape $output";
	#SelfSystem($command);
	if ($DEBUG) {
	    print "$command\n";
	}
	else {
	    system($command);
	}

	$ctr = $ctr + 1;
    }
    print "\n";
    $i += 1;
}
