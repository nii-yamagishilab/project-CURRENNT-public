#!/usr/bin/perl
use Cwd;
use File::Basename;

if (@ARGV<1){
    die " **** Usage: perl 00_RUN.pl config.pm\n";
}
require("$ARGV[0]") || die "Can't load $config";
$config = $ARGV[0];

if ($DEBUG==1){
    print_time("Entering DEBUG mode");
}




if ($FLAG_SCP || $PREP_DAT){
    print_time("Start preparing the training data");
    my $j = 0;
    foreach my $i (@datadir){
	SelfSystem("perl 01_prepare_train_data.pl $i $buffdir[$j] $config");
	$j += 1;
    }
}else{
    print_time("Skip preparing the training data");
}


if ($TRAINSYS){
    print_time("Training the model");
    my $j = 0;
    my $curdir = getcwd;
    foreach my $i (@datadir){
	my $prj = $sysdir[$j];
	if (-e $prj){
	}else{
	    mkdir $prj;
	}
	if ($useOldConfig){
	}else{
	    print "Old config.cfg and network cleared\n";
	    print "   remember to set $useOldConfig=0 after";
	    print "   you craft new config.cfg and network.jsn";
	    SelfSystem("rm $prj/config.cfg");
	    SelfSystem("rm $prj/network.jsn");
	}
	if (-e "$prj/config.cfg" && -e "$prj/network.jsn"){
	    chdir($prj);
	    print `pwd`;
	    SelfSystem("$currennt --options_file config.cfg > log_train 2>&1");
	    chdir($curdir);
	}else{
	    SelfSystem("python 02_prepare_model.py $prj $buffdir[$j]");
	    print "Please cd $prj\n";
	    print "  modify the $prj/config.cfg_tmp \n  and $prj/network.jsn_tmp\n";
	    print "  then change the name (delete _tmp) and run again\n";
	}
	$j += 1;
    }
}else{
    print_time("Skip model traininig");
}


if ($TEST_FLAG_SCP || $TEST_PREP_DAT ) {
    print_time("Start preparing the test data");
    my $i = 0;
    unless(scalar @testdatadir == scalar @mvpath){
	die "testdatadir is not equal in length as \@mvpath";
    }
    unless(scalar @testdatadir == scalar @testbuffdir) {
	die "testdatadir is not equal in length as \@testbuffdir";
    }
    foreach $name (@testdatadir){
	my $dir = "$name";
	SelfSystem("perl 03_prepare_test_data.pl $dir $mvpath[$i] $testbuffdir[$i] $config");
	$i += 1;
    }
}else{
    
    print_time("Skip preparing the test data");
}




if ($GENDATA || $SPLITDATA || $SYNWAVE){
    print_time("Start synthesizing the speech");
    if (scalar @testdatadir == scalar @networkdir){
    }else{
	die "testdatadir is not equal in length as \@networkdir";
    }
    if (scalar @testdatadir == scalar @testbuffdir){
    }else{
	print "@testdatadir\n";
	print "@testbuffdir\n";
	die "testdatadir is not equal in length as \@testbuffdir";
    }
    if (scalar @testdatadir == scalar @sysdir){
    }else{
	die "testdatadir is not equal in length as \@sysdir";
    }
    if (scalar @borrowdir == 0 || scalar @testdatadir == scalar @borrowdir){
    }else{
	#print "\n";
	print "testdatadir is not equal in length as \@borrowdir\n";
	die "set \@borrowdir=(); if you don't know how to use it";
    }
    my $i = 0;
    foreach my $temp (@testdatadir){
    if ($i<@testdatadir){
	my $dataName  = $testdatadir[$i]; # where is the data_config.py and gen.scp
	my $tmp       = $networkdir[$i];  # 
	my @netNames  = @$tmp;            # the names of the nets for generating features
	my $sysName   = $sysdir[$i];      # where are the nets

	my @borDir;                       # 
	if (scalar @borrowdir > 0){
	    my $tmp      = $borrowdir[$i];
	    @borDir   = @$tmp;
	}else{
	    @borDir   = ();
	}
	my @spgDir, @lf0Dir, @bapDir;
	if (scalar @SPGDir > 0){
	    my $tmp      = $SPGDir[$i];
	    @spgDir   = @$tmp;
	}else{
	    @spgDir   = ();
	}
	if (scalar @LF0Dir > 0){
	    my $tmp      = $LF0Dir[$i];
	    @lf0Dir   = @$tmp;
	}else{
	    @lf0Dir   = ();
	}
	if (scalar @BAPDir > 0){
	    my $tmp      = $BAPDir[$i];
	    @bapDir   = @$tmp;
	}else{
	    @bapDir   = ();
	}
	
	opendir(DIR, $testbuffdir[$i]);
	my @ncDataList = readdir(DIR);
	@ncDataList = grep(/data[a-z0-9]*.nc[0-9a-z]*/, @ncDataList);
	close(DIR);
	print "Data to be generated @ncDataList\n";
	
	# for each system
	my $j = 0;
	foreach my $netName (@netNames){
	    my $tmp = $netName; $tmp =~ s/\.[a-zA-Z0-9]+$//; # basename
	    my $odir  = sprintf("%s/%s_%s", $sysName, $outputNameBase, $tmp);  # output dir
	    my $bdir, $spgdir, $lf0dir, $bapdir;
	    
	    # check borrow directory
	    if (scalar @borDir==0){
		$bdir  = $odir;
	    }else{
		scalar @borDir == scalar @netNames || die "Uequal length of borDir and net";
		$bdir  = sprintf("%s/%s", $sysName, $borDir[$j]);      
	    }
	    # check spg, lf0 and bap directory for output
	    if (scalar @spgDir==0){
		$spgdir  = $odir;
	    }else{
		scalar @spgDir == scalar @netNames || die "Uequal length of spgDir and net";
		$spgdir  = $spgDir[$j];       
		if ($pgflag){
		    print "MGC will be enhanced again using postfilter\n";
		}
	    }
	    if (scalar @lf0Dir==0){
		$lf0dir  = $odir;
	    }else{
		scalar @lf0Dir == scalar @netNames || die "Uequal length of spgDir and net";
		$lf0dir  = $lf0Dir[$j];       
	    }
	    if (scalar @bapDir==0){
		$bapdir  = $odir;
	    }else{
		scalar @bapDir == scalar @netNames || die "Uequal length of spgDir and net";
		$bapdir  = $bapDir[$j];       
	    }
	    
	    my $nname = sprintf("%s/%s", $sysName, $netName); 
	    if ($flagCLEAN){
		SelfSystem("rm $odir/*");
	    }
	    if (-e "$nname"){
		if (-e "$ordir"){
		}else{
		    mkdir $odir;
		}

		my $flag1 = $GENDATA;
		my $flag2 = $SPLITDATA;
		my $flag3 = $SYNWAVE;
		
		# only iterate over @ncDataList
		$SPLITDATA=0;
		$SYNWAVE=0;
		foreach my $ncfile (@ncDataList){
		    $command = "perl 04_synwav.pl $dataName $testbuffdir[$i]/$ncfile";
		    $command = "$command $nname $sysName $odir $bdir $nndata $config";
		    $command = "$command $spgdir $lf0dir $bapdir $pgflag";
		    SelfSystem($command);
		    print "Generating features to $odir\n";
		}
		
		# generate the data and .wav
		$GENDATA = 0;
		$SPLITDATA=$flag2;
		$SYNWAVE=$flag3;
		$command = "perl 04_synwav.pl $dataName DUMMY";
		$command = "$command $nname $sysName $odir $bdir $nndata $config";
		$command = "$command $spgdir $lf0dir $bapdir $pgflag";
		SelfSystem($command);
		print "Generating features to $odir\n";
		$GENDATA=$flag1;

	    }else{
		print "Ignore: $nname\n";
	    }   
	    $j += 1;
	}
    }
    $i += 1;
    }
}else{
    print_time("Skip parameter generating");
}




if($CALRMSE){
    print_time("Calculating RMSE");
    my $i=0;
    SelfSystem("rm ./tmp/*");
    foreach my $temp (@RMSEFeatDir){
	if ($i < scalar @RMSEFeatDir){
	    my $name = $RMSEFeatDir[$i];

	    defined ($jobID = fork) or die 'Cannot fork: $!';
	    unless($jobID){
		@tmpname = split('/',$name);
		$tmpname = $tmpname[@tmpname-2];
		$command = "perl 05_calRMSE.pl $name $tmpname $i $config > ./tmp/RMSE-$i.log";
	
		if ($DEBUG){
		    print $command,"\n";
		    die "Finish printing";
		}else{
		    exec($command);
		}
		#die "Finish $outdir";
	    }
	    $jobCount += 1;
	    $nJobs[$jobCount] = $jobID;
	    if ($jobCount == $maxPara){
		foreach $job (@nJobs){
		    waitpid($job,0);
		}
		$jobCount = 0;
		@nJobs = ();
	    }
	}
	$i += 1;
    }
    foreach $job (@nJobs){
	waitpid($job,0);
    }
    my $i = 0;
    print "Print RMSE information\n";
    if ($DEBUG==0){
	while($i < scalar @RMSEFeatDir){
	    system("cat ./tmp/RMSE-$i.log");
	    $i += 1;	    
	}
    }
}else{
    print_time("Skip calculating RMSE");
}
