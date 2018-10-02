#!/usr/bin/perl
use Cwd;
use File::Basename;


# Argument check
if (@ARGV<1){
    die " **** Usage: perl 00_RUN.pl config.pm\n";
}
require("$ARGV[0]") || die "Can't load $config";
$config = $ARGV[0];

if ($ARGV[1] > 0){
    #print "DEBUG mode\n";
    $DEBUG = 1;
}


if ($DEBUG==1){
    print_time("Entering DEBUG mode");
}

# Preparing the data
if ($FLAG_SCP || $PREP_DAT){
    my $curdir = getcwd;
    print_time("------- 1 Start preparing the training data -------");
    my $j = 0;
    foreach my $i (@datadir){
	SelfSystem("perl 01_prepare_train_data.pl $i $buffdir[$j] $config");
	$j += 1;
    }
}else{
    print_time("------- 1 Skip preparing the training data -------");
}

# Train the system
if ($TRAINSYS){
    print_time("------- 2 Training the model -------");
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
	    SelfSystem("rm -f $prj/config.cfg");
	    SelfSystem("rm -f $prj/network.jsn");
	}
	if (-e "$prj/config.cfg"){
	    chdir($prj);
	    print `pwd`;
	    SelfSystem("cat ./config.cfg");
	    print "\n\tModel training. Please wait for several minutes. \n";
	    SelfSystem("$currennt --options_file config.cfg > log_train 2>&1");
	    print "\n-------------------- CURRENNNT training log ----------------\n";
	    SelfSystem("cat ./log_train");
	    print "------------------------------------------------------------\n";
	    chdir($curdir);
	}else{
	    die "Can't find config.cfg in $prj\n";
	    #SelfSystem("python 02_prepare_model.py $prj $buffdir[$j]");
	    #print "Please cd $prj\n";
	    #print "  modify the $prj/config.cfg_tmp \n  and $prj/network.jsn_tmp\n";
	    #print "  then change the name (delete _tmp) and run again\n";
	}
	$j += 1;
    }
}else{
    print_time("------- 2 Skip model traininig -------");
}

# Generating the test data based on data.nc
if ($TEST_FLAG_SCP || $TEST_PREP_DAT ) {
    print_time("------- 3 Start preparing the test data -------");
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
    
    print_time("------- 3 Skip preparing the test data -------");
}


# Generating the acoustic features
# $CALRMSE is included to generate the @RMSEFeatDir
if ($GENDATA || $SPLITDATA || $SYNWAVE || $CALRMSE){
    
    if ($CALRMSE){
	if (scalar @RMSEFeatDir > 0){
	    print "No need to specify RMSEFeatDir anymore, they will be inferred\n";
	}
	@RMSEFeatDir = ();
    }
    if ($GENDATA || $SPLITDATA || $SYNWAVE){
	print_time("------- 4 Start synthesizing the speech -------");
    }else{
	print_time("------- 4 Get output dirs -------");
    }

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
	
	my $mvfile    = $mvpath[$i];      
	if ($useDataMV){
	    $mvfile   = $mvpath[$i];
	}else{
	    $mvfile   = 'NONE';
	}
	
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
	    my $bdir, $spgdir, $lf0dir, $bapdir;
	    my $odir;
	    my $weCommand = "NONE NONE NONE";
	    if (defined $weExt && $weExt){
		#print "Using External enbedded vectors \n";
		if (defined $weDim && defined $weIDDim){
		    print "Using External enbedded vectors \n";		
		    my $tmp = $wedir[$i];
		    my @we  = @$tmp;
		    if (scalar @we > 0){
			scalar @we == scalar @netNames || die "Une we,netNames @we @netNames \n";
			if (-e "$we[$j]"){
			    $weCommand = "$we[$j] $weIDDim $weDim";
			}else{
			    $weCommand = "$sysName/$we[$j] $weIDDim $weDim";
			}
		    }else{
			# not use we
		    }
		}else{
		    #die "weIDDim weDim and wedir not defined\n";
		    #not use we
		}
	    }

	    # output directory
	    if (defined $weExt && $weExt){
		my $tmp = $wedir[$i];
		@we  = @$tmp;
		if (scalar @we > 0){
		    $tmp = $we[$j]; 
		    $tmp = `basename $tmp`;         # we name
		    $tmp =~ s/\.[a-zA-Z0-9\.\n]+//; # basename
		    chomp($tmp);
		    $tmp2 = `basename $netName`; $tmp2 =~ s/\.[a-zA-Z0-9\n]+$//; # basename
		    $odir  = sprintf("%s/%s_%s_we%s", $sysName, $outputNameBase, $tmp2, $tmp);  
		}else{
		    my $tmp = $netName; $tmp =~ s/\.[a-zA-Z0-9]+$//; # basename
		    $odir  = sprintf("%s/%s_%s", $sysName, $outputNameBase, $tmp);  # output dir
		}	
	    }else{
		my $tmp = $netName; $tmp =~ s/\.[a-zA-Z0-9]+$//; # basename
		$odir  = sprintf("%s/%s_%s", $sysName, $outputNameBase, $tmp);  # output dir
	    }
	    
	    my $tapCommand = "NONE NONE";
	    if (defined $tapOutput && $tapOutput > 0){
		if (scalar @tapLayer > $i){
		    my $tapLayer = $tapLayer[$i];
		    my $tapgate  = $tapGate[$i];
		    $tapCommand = "$tapLayer $tapgate";
		    $odir = sprintf("%s_%s_%s", $odir, $tapLayer, $tapgate);
		}else{
		    print "Invalid tap output specification, not use tap\n";
		}
	    }
	    
	    my $mdnGenCommand = "NONE";
	    if (@mdnGenPara){
		if (scalar @mdnGenPara > $i){
		    my $tmpmdnGenPara = $mdnGenPara[$i];
		    if ($tmpmdnGenPara eq "NONE"){

		    }
		    elsif ($tmpmdnGenPara > 0.0 ){
			$odir = sprintf("%s_mdn%f", $odir, $tmpmdnGenPara);
			$mdnGenCommand = "$tmpmdnGenPara";
		    }elsif ($tmpmdnGenPara > -2.0){
			$odir = sprintf("%s_mdnPara", $odir);
			$mdnGenCommand = "$tmpmdnGenPara";
		    }else{
			$odir = sprintf("%s_mdnRawPara", $odir);
			print "Ignore this infor if the network is not MDN: \n";
			print "\tBy default, for MDN, raw MDN parameter without ";
			print "transformation will be generated\n";
		    }
		}
	    }

	    if (1){
		# check borrow directory
		if (scalar @borDir==0){
		    $bdir  = $odir;
		}else{
		    scalar @borDir == scalar @netNames || die "Uequal length of borDir and net";
		    $bdir  = sprintf("%s/%s", $sysName, $borDir[$j]);      
		}
		# check spg, lf0 and bap directory for output
		unless (defined $pgflag){
		    $pgflag = 1;
		}

		# check any pre-generated acoustic features
		if (scalar @spgDir==0){
		    $spgdir  = $odir;
		}else{
		    scalar @spgDir == scalar @netNames || die "Uequal length of spgDir and net";
		    $spgdir  = $spgDir[$j];       
		    if ($pgflag){print "MGC will be enhanced again using postfilter\n";}	
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
	    
		my $nname;
		if (-e "$netName"){
		    # full path
		    $nname = $netName;
		}else{
		    # relative path
		    $nname = sprintf("%s/%s", $sysName, $netName); 
		}  
	    
		if (-e "$nname"){
		    
		    if ($CALRMSE) {
			push @RMSEFeatDir, $odir;
		    }
		    if ($GENDATA || $SPLITDATA || $SYNWAVE){
			if ($flagCLEAN){SelfSystem("rm -f $odir/*");}
			unless(-e "$ordir"){mkdir $odir;}
			# only iterate over @ncDataList
			foreach my $ncfile (@ncDataList){
			    $command = "perl 04_synwav.pl $dataName $testbuffdir[$i]/$ncfile";
			    $command = "$command $nname $sysName $odir $bdir $nndata $config";
			    $command = "$command $spgdir $lf0dir $bapdir $pgflag 1 0 0 NONE";
			    $command = "$command $weCommand $tapCommand $mdnGenCommand";
			    SelfSystem($command);
			    print "Generating features to $odir\n";
			}
			# generate the data and .wav
			$command = "perl 04_synwav.pl $dataName DUMMY";
			$command = "$command $nname $sysName $odir $bdir $nndata $config";
			$command = "$command $spgdir $lf0dir $bapdir $pgflag 0 1 1 $mvfile";
			$command = "$command NONE NONE NONE NONE NONE NONE";
			SelfSystem($command);
			print "Generating features to $odir\n";
		    }
		}else{
		    print "Ignore: $nname\n";
		}  
	    } # generate the acoustic features and waves
	    
	    $j += 1;
	}
    }
    $i += 1;
    }
}else{
    print_time("------- 4 Skip parameter generating -------");
}




if($CALRMSE){
    print_time("------- 5 Calculating RMSE -------");
    my $i=0;
    SelfSystem("mkdir ./tmp");
    SelfSystem("rm -f ./tmp/*");
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
    print_time("------- 5 Skip calculating RMSE -------");
}

print_time(" --- All Done --- ");
print "$CONFIG_ADDINFO\n";
