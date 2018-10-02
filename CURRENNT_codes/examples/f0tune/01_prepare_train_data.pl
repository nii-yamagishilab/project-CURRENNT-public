#!/usr/bin/perl
use Cwd;
use File::Basename;
###############################################################
## General Description:
##		This script prepares the input data to CURRENT
## Usage:
##		steps:
##		1. Given the randomized list of all the features (input and output)
##		   in $randomDir, extract $numUtt utterances and print
##		   the lists to the folder $prjdir/filename
##		2. Given the lists of all features and the data_config.py
##		   in $prjdir, accumulate the information about the dimension
##		   and timesteps of the utterances, and generate $prjdir/all.scp
##		3. Given $prjdir/all.scp, put all the data into multiple .nc data files,
##		   normalize the data, and output the mean and std to data.mv
##              4. If VAL_DAT = 1, $valUtt utterances will be separated as validation
##                 set from the last .nc files. This process can be lauched after
##                 .nc have been generated (post step). Or you can manually set one .scp
##                 for validation set and directly genereate in step 3
## Notes:
##		Please configure data_config.py before running
###############################################################

$prjdir =  $ARGV[0];    # ""
$buffadd = $ARGV[1];    # the buffer directory, .nc data will be stored here
                        # $buffadd should be a local disk, not NFS path
                        # otherwise, it is extremly slow

require($ARGV[2]) || die "Can't find config.pm";

# reserved options. No need to modify them
$randomDir = "/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/random_list";
$numUtt      = 13000;    # number of utterance to be included
                         # if larger than total number, it will be total number
$valUtt      = 500;      # number of validation set
$flagMultiNC = 1;        # whether there are multiple nc files ?
$RAND_SCP = 0;           # Reserved
$VAL_DAT  = 0;           # Reserved

mkdir $buffadd, 0755;

if ( -e "$prjdir/data_config.py" ) {
    print "Using $prjdir/data_config.py\n";
}
else {
    print "Found no $prjdir/data_config.py\n";
    die "Please use ./template/data_config.py_train as template\n";
    #die "";
}

if ($RAND_SCP) {
    print "Found the randomized list in $randomDir\n";
    # generate .scp
    opendir( $D, $randomDir ) || die "can't open $randomDir";
    @files = readdir($D);
    @files = grep { /[\S]+.scp/ } @files;
    closedir(D);

    foreach $file (@files) {
        system("echo $file");
        $filename = $file;
        $filename =~ s/\_r\_full//g;
	$command = "cat $randomDir/$file | head -n $numUtt > $prjdir/$filename";
        SelfSystem($command);
    }

}

if ($FLAG_SCP) {
    # Delete the old *.info if new .scp is utilized
    if (-e "$prjdir/data_config.py"){
    }else{
	die "Can't find $prjdir/data_config.py";
    }
    SelfSystem("rm $prjdir/*.info");
    $command = "python ./utilities/PrePareData.py $prjdir/data_config.py $prjdir";
    SelfSystem($command);

}

if ($PREP_DAT) {
    if ( -e "$prjdir/all.scp" ){
    }else{
	die "Can't find $prjdir/all.scp";
    }
    
    if ( -e "$prjdir/mask.txt" ) {
        print "Using the mask file $prjdir/mask.txt\n";
        $maskfile = "$prjdir/mask.txt";
    }
    else {
        $maskfile = "";
        print "Not using mask";
    }
    
    if ($dataPack ne ""){
	$command = "python ./utilities/PackData.py $prjdir/all.scp $prjdir/data.mv";
	if ($maskfile ne ""){
	    $command = "$command $buffadd 1 1 1 $maskfile";
	}else{
	    $command = "$command $buffadd 1 1 1 None";
	}
	SelfSystem($command);
    }else{

    if ($flagMultiNC) {
        $scpFile = "$prjdir/all.scp";
        $dataScp = "$prjdir/data.scp";
        open( IN_STR,  "$scpFile" );
        open( OUT_STR, ">$dataScp" );
        $count = 1;
        while (<IN_STR>) {
            chomp();
            $file = $_;
            $name = basename($file);
            $name =~ s/all.scp/data.nc/g;
            if ( $maskfile eq "" ) {
                $commandline = "$bmat2nc $file $buffadd/$name";
            }
            else {
                $commandline = "$bmat2nc $file $buffadd/$name $maskfile";
            }
            print "$commandline\n";
            SelfSystem($commandline);
            print OUT_STR "$buffadd/$name\n";
            $count = $count + 1;
        }
        close(IN_STR);
        close(OUT_STR);

        $commandline = "$ncnorm $dataScp + $prjdir/data.mv";

        #print "$commandline\n";
        SelfSystem($commandline);

        $count = $count - 1;
        open( IN_STR, "$dataScp" );
        while (<IN_STR>) {
            chomp();
            $commandline = "$ncnorm $_ $prjdir/data.mv";

            #print "$commandline\n";
            SelfSystem($commandline);
            $count = $count - 1;
        }
    }
    }
}

if ($VAL_DAT) {
    if ( -e "$prjdir/all.scp" ) {
        open( IN_STR, "$prjdir/all.scp" );
        $lastScp = "";
        while (<IN_STR>) {
            chomp();
            $lastScp = $_;
        }
        close(IN_STR);
        if ( -e "$lastScp" ) {
            $num = $lastScp;
            $num =~ m/\.scp([0-9]+)/;
            $num = $1;
            open( IN_STR, "$lastScp" );
            $tempscp1 = "$lastScp" . "val";
            $tempscp2 = "$lastScp" . "train";
            open( OUT_STR_1, ">$tempscp1" );
            open( OUT_STR_2, ">$tempscp2" );

            while (<IN_STR>) {
                chomp();
                if ( $valUtt > 0 ) {
                    print OUT_STR_1 "$_\n";
                    $valUtt -= 1;
                }
                else {
                    print OUT_STR_2 "$_\n";
                }
            }
            close(OUT_STR_1);
            close(OUT_STR_2);

            if ( -e "$prjdir/mask.txt" ) {
                print "Using the mask file $prjdir/mask.txt\n";
                $maskfile = "$prjdir/mask.txt";
            }
            else {
                $maskfile = "";
            }

            if ( $maskfile eq "" ) {
                $commandline =
                  "$bmat2nc $tempscp1 $prjdir/data.nc" . "$num" . "val";
                SelfSystem($commandline);
                $commandline =
                  "$bmat2nc $tempscp2 $prjdir/data.nc" . "$num" . "train";
                SelfSystem($commandline);
            }
            else {
                $commandline = "$bmat2nc $tempscp1 $prjdir/data.nc" . "$num"
                  . "val $maskfile";
                SelfSystem($commandline);
                $commandline = "$bmat2nc $tempscp2 $prjdir/data.nc" . "$num"
                  . "train $maskfile";
                SelfSystem($commandline);

            #$commandline = "$bmat2nc $file $prjdir/data.nc"."$count $maskfile";
            }

            #print OUT_STR "$prjdir/data.nc"."$count\n";
            if ( -e "$prjdir/data.mv" ) {
                $commandline =
                  "$ncnorm $prjdir/data.nc" . "$num" . "val $prjdir/data.mv";
                SelfSystem($commandline);
                $commandline =
                  "$ncnorm $prjdir/data.nc" . "$num" . "train $prjdir/data.mv";
                SelfSystem($commandline);

            }
            else {
                print "Can't find data.mv. Data will not be normalized\n";
            }

        }
        else {
            print "Can't find $lastScp\n";
        }
    }
    else {
        print "can't find $prjdir/all.scp. Can't generate validation set\n";
    }
}

