#!/usr/bin/perl

# Utils ==============================
sub shell($) {
   my ($command) = @_;
   my ($exit);

   $exit = system($command);

   if ( $exit / 256 != 0 ) {
      die "Error in $command\n";
   }
}

sub print_time ($) {
   my ($message) = @_;
   my ($ruler);

   $message .= `date`;

   $ruler = '';
   for ( $i = 0 ; $i <= length($message) + 4 ; $i++ ) {
      $ruler .= '=';
   }

   print "\n$ruler\n";
   print "@_ at " . `date`;
   print "$ruler\n\n";
}

sub get_file_size($) {
   my ($file) = @_;
   my ($file_size);

   $file_size = `$WC -c < $file`;
   chomp($file_size);

   return $file_size;
}

sub path_wo_ext($$) {
   my ($path, $ext) = @_;
   my ($bname, $dname);

   $bname = `basename $path $ext`;
   chomp($bname);
   $dname = `dirname $path`;
   chomp($dname);
   $path = $dname . "/" . $bname;

   return $path;
}

sub set_speaker_files($$) {
   my ( $file, $ext ) = @_;

   $spkr = `dirname $file`;
   chomp($spkr);
   $spkr = `basename $spkr`;
   chomp($spkr);
   $base = `basename $file $ext`;
   chomp($base);

   $raw  = "$datdir/raw/$spkr/$base.raw";
   $sp   = "$datdir/sp/$spkr/$base.sp";
   $mgc  = "$datdir/mgc/$spkr/$base.mgc";
   $f0   = "$datdir/f0/$spkr/$base.f0";
   $lf0  = "$datdir/lf0/$spkr/$base.lf0";
   $ap   = "$datdir/ap/$spkr/$base.ap";
   $bap  = "$datdir/bap/$spkr/$base.bap";
   $cmp  = "$datdir/cmp/$spkr/$base.cmp";
   $txt  = "$datdir/txt/$spkr/$base.txt";
   $full = "$datdir/labels/full/$spkr/$base.lab";
   $mono = "$datdir/labels/mono/$spkr/$base.lab";

   $wpsp     = "$nndatdir/sp/$spkr/$base.sp_warp";
   $exsilsp  = "$nndatdir/sp_exsil/$spkr/$base.exsil.sp_warp";
   $mgc_d    = "$nndatdir/mgc_delta/$spkr/$base.mgc";
   $vuv      = "$nndatdir/vuv/$spkr/$base.vuv";
   $iplf0    = "$nndatdir/iplf0/$spkr/$base.lf0";
   $iplf0_d  = "$nndatdir/iplf0_delta/$spkr/$base.lf0";
   $bap_d    = "$nndatdir/bap_delta/$spkr/$base.bap";
   $full_a   = "$nndatdir/labels/full_align/$spkr/$base.lab";
   $lab      = "$nndatdir/lab/$spkr/$base.lab";

   $ae      = "$aedatdir/ae/$spkr/$base.ae";

   $iaednnsp      = "$pfdatdir/train/$spkr/$base.sp_warp";
   $exsiliaednnsp = "$pfdatdir/train/$spkr/$base.exsil.sp_warp";
}

# PBS ==============================
sub make_job($) {
   my ( $jobver ) = @_;

   open( JOB, ">$jobdir/job.sh" ) || die "Cannot open $!";
   print JOB "\#\!/bin/sh\n\n";
   if ( ! $jobver ) {
      print JOB "/usr/bin/perl $train $config $utils \$jobflag \$jobopt1 \$jobopt2 \$jobopt3 > $logdir/job.\$jobnum.\$jobflag.\$jobopt1.\$jobopt2.\$jobopt3.log 2> $logdir/job.\$jobnum.\$jobflag.\$jobopt1.\$jobopt2.\$jobopt3.error.log \n";
   }
   else {
      print JOB "/usr/bin/perl $train $config $utils \$jobflag \$jobopt1 \$jobopt2 \$jobopt3 > $logdir/job.$jobver.\$jobnum.\$jobflag.\$jobopt1.\$jobopt2.\$jobopt3.log 2> $logdir/job.$jobver.\$jobnum.\$jobflag.\$jobopt1.\$jobopt2.\$jobopt3.error.log \n";
   }
   close(JOB);
   shell("chmod +x $jobdir/job.sh");
}

sub set_jobopts($$$) {
   my ( $opt1, $opt2, $opt3) = @_;

   $jobopt1 = $opt1;
   $jobopt2 = $opt2;
   $jobopt3 = $opt3;
}

sub pbs_qsub($$$$$$) {
   my ( $cput, $mem, $cmd ) = @_;
   my ( $opt );

   if ($USEPBS) {
      if ( $jobopt1 !~ /@/ && $jobopt2 !~ /@/ && $jobopt3 !~ /@/) {
         $jobname = "job.$jobnum.$jobflag.$jobopt1.$jobopt2.$jobopt3";
         $opt     = "-o $blogdir/$jobname.log -N $jobname -v jobnum=$jobnum,jobflag=$jobflag,jobopt1=$jobopt1,jobopt2=$jobopt2,jobopt3=$jobopt3";
         
         $jobID = `qsub -cwd -o /dev/null -e /dev/null -l $cput -P inf_hcrc_cstr_udialogue -pe memory-2G $mem $opt $cmd`;
         $chr   = `echo $?`;
         $ne    = 0;
         while ( $chr != 0 ) {
            $ne++;
            if ( $ne == 5 ) {
               print "error: Could not qsub\n";
               exit(0);
            }
            sleep(10);
            $jobID = `qsub -cwd -o /dev/null -e /dev/null -l $cput -P inf_hcrc_cstr_udialogue -pe memory-2G $mem $opt $cmd`;
            $chr   = `echo $?`;
         }
      }
      else {
         shell("export jobnum=$jobnum jobflag=$jobflag jobopt1=$jobopt1 jobopt2=$jobopt2 jobopt3=$jobopt3; $cmd");
      }
   }
   else {
      defined ($jobID = fork) or die "Cannot fork: $!";
      unless($jobID) {
         exec("export jobnum=$jobnum jobflag=$jobflag jobopt1=$jobopt1 jobopt2=$jobopt2 jobopt3=$jobopt3; $cmd");
         die "cannot exec: $!";
      }
   }

   $jobID = `echo "$jobID" | cut -d ' ' -f 3`;
   chomp($jobID);
   return $jobID;
}

sub pbs_wait($) {
   my ( $jobver ) = @_;

   if ($USEPBS) {
      $failnum = 0;
      do {
         $qstat = `qstat | sed 1,2d`;
         $chr   = `echo $?`;
         $flag  = 0;
         if ( $chr == 0 ) {
            $failnum = 0;
            foreach $jobID (@Jobs) {
               chomp($jobID);
               if ( ( $jobID ne "" && $qstat =~ /$jobID/ ) || $qstat =~ /No Permission/ || $qstat =~ /Cannot/ ) {
                  $flag = 1;
                  last;
               }
            }
         }
         else {
            $failnum++;
            $flag = 1;
            if ( $failnum == 5 ) {
               print "error: Could not qstat\n";
               exit(0);
            }
         }
         sleep(30);
      } while ($flag);
   }
   else {
      foreach $jobID (@Jobs) {
         waitpid($jobID, 0);
      }
   }

   check_pbs($jobver);
   @Jobs = ();
}

sub check_pbs($) {
   my ( $jobver ) = @_;

   if ($USEPBS) {
      @ERROR = ( 'ERROR', 'Error', 'error', 'Terminated' );
      foreach $error (@ERROR) {
         if ( ! $jobver ) {
            $nlog  = `grep $error $logdir/job.$jobnum.$jobflag.*.error.log | $WC -l`;
            $nblog = `grep $error $blogdir/job.$jobnum.$jobflag.*.log | $WC -l`;
         }
         else {
            $nlog  = `grep $error $logdir/job.$jobver.$jobnum.$jobflag.*.error.log | $WC -l`;
            $nblog = `grep $error $blogdir/job.$jobver.$jobnum.$jobflag.*.log | $WC -l`;
         }
         if ( $nlog > 0 || $nblog > 0 ) {
            print "error: check_pbs $error exit\n";
            exit(0);
         }
      }
   }
   else {
      @ERROR = ( 'ERROR', 'Error', 'Terminated' );
      foreach $error (@ERROR) {
         if ( ! $jobver ) {
            $nlog  = `grep $error $logdir/job.$jobnum.$jobflag.*.error.log | $WC -l`;
         }
         else {
            $nlog  = `grep $error $logdir/job.$jobver.$jobnum.$jobflag.*.error.log | $WC -l`;
         }
         if ( $nlog > 0 ) {
            print "error: check_pbs $error exit\n";
            exit(0);
         }
      }
   }
}

# lists ==============================
sub make_parallel_scp($$$) {
   my ( $scp, $nPara, $p ) = @_;
   my ( $file, $i );

   $i = 0;
   open( SCP,     "cat $scp |" ) || die "Cannot open $!";
   open( SCPPARA, ">$scp.$p" )   || die "Cannot open $!";
   while ( $file = <SCP> ) {
      chomp($file);
      $i = $i % $nPara + 1;
      if ( $i == $p ) {
         print SCPPARA "$file\n";
      }
   }
   close(SCPPARA);
   close(SCP);
}

sub compose_files($$$) {
   my ($file, $nPara) = @_;
   my ($p);
   
   shell("rm -f $file");
   for ( $p = 1; $p <= $nPara; $p++ ) {
      if ( -s "$file.$p" ) {
         shell("cat $file.$p >> $file");
      }
      shell("rm -f $file.$p");
   }
   if ( -s "$file" ) {
      shell("sort -u -o $file $file");
   }
}

# Features ==============================
sub lf02f0($$) {
   my ($input, $output) = @_;

   shell("$SOPR -magic -1.0E+10 -d 1127 -EXP -s 1 -m 700 -MAGIC 0.0 $input > $output");
}

sub bap2ap($$) {
   my ($input, $output) = @_;

   shell( "$BNDAP2AP $input > $output" );
}

# text ==============================
sub preprocessing_text_analysis($$) {
   my ($input, $output) = @_;

   $cmd = "tr -d '\^\"\`\?\!\(\)\*\_\+\=\:\[\]\|\\\\~\<\>\;\/' < $input | ";
   $cmd .= "sed \"s/ \'/ /g\" | sed \"s/\' / /g\" | sed 's/\\. / /g' | sed 's/\- / /g' | sed 's/ \-/ /g' | ";
   $cmd .= "sed 's/  \*/ /g' | sed 's/^  \*//g' | sed 's/  \*\$//' | ";
   $cmd .= "sed \"s/^'//g\" | sed \"s/\'\$//g\" | ";
   $cmd .= "$PERL -pe 's/\n/ /g' | $PERL -pe 's/\\.+//g' | $PERL -pe 's/,+/,/g' > $output";
   shell($cmd);
}

# Synthesis ==============================
sub gen_wave($$$$$$) {
   my ( $gendir, $spgendir, $f0gendir, $apgendir, $scp, $ext, $useSigPF2, $useMLPG, $synWav) = @_;
   my ( $tspgendir, $tf0gendir, $tapgendir);
   my ( $line, @FILE, $file, $base, $T, $dim );

   $tspgendir = $spgendir;
   $tf0gendir = $f0gendir;
   $tapgendir = $apgendir;

   @FILE = split( '\n', `cat $scp` );
   $lgopt = "-l" if ($lg);

   print "Processing directory $gendir:\n";

   foreach $file (@FILE) {
      $spgendir = $tspgendir;
      $f0gendir = $tf0gendir;
      $apgendir = $tapgendir;

      $base = `basename $file $ext`;
      chomp($base);
      if (-s "$gendir/$base.wav"){
	  print "$gendir/$base.wav is there\n";
	  next;
      }
      if ( -s "$spgendir/$base.mgc_delta" ) {
      	 if ($useMLPG){
	     $T = get_file_size("$spgendir/$base.mgc_delta") / (4*$ordr{'mgc'}*$nwin{'mgc'});
	     shell("rm -f $gendir/$base.mgc_delta.var");
	     for ( $i = 1 ; $i <= $T ; $i++ ) {
		 shell("cat $nndatdir/var/mgc.var >> $gendir/$base.mgc_delta.var");
	     }
	     $dim = $ordr{'mgc'} * $nwin{'mgc'};
	     shell("$MERGE -s 0 -l $dim -L $dim $spgendir/$base.mgc_delta < $gendir/$base.mgc_delta.var > $gendir/$base.mgc_pdf");
	     shell("$MLPG -l $ordr{'mgc'} -d $nndatdir/win/mgc.win2.f -d $nndatdir/win/mgc.win3.f $gendir/$base.mgc_pdf > $gendir/$base.mgc");
	     $spgendir = "$gendir";
         }else{
	     # just rename the .mgc_delta as .mgc
	     shell("mv $gendir/$base.mgc_delta $gendir/$base.mgc");
         }
      }
      
      if ($synWav && -s "$spgendir/$base.mgc" ) {
	  if ( $useSigPF && $useSigPF2 ) {
	      shell( "$MCPF -a $fw -m " . ( $ordr{'mgc'} - 1 ) . " -b $pf_mcp -l $il $spgendir/$base.mgc > $gendir/$base.p_mgc" );
	      $mgc = "$gendir/$base.p_mgc";
	  }
	  else {
	      $mgc = "$spgendir/$base.mgc";
	  }

         if ( $gm == 0 ) {
            shell( "$MGC2SP -a $fw -g $gm -m " . ( $ordr{'mgc'} - 1 ) . " -l $fp -o 2 $mgc > $gendir/$base.sp" );
         }
         else {
            $line = "$LSPCHECK -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " -c -r 0.1 $mgc | ";
            $line .= "$LSP2LPC -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " $lgopt | ";
            $line .= "$MGC2MGC -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw -c $gm -n -u -M " . ( $ordr{'mgc'} - 1 ) . " -A $fw -C $gm | ";
            $line .= "$MGC2SP -a $fw -c $gm -m " . ( $ordr{'mgc'} - 1 ) . " -l $fp -o 2 > $gendir/$base.sp";
            shell($line);
         }         
      }
      if ( -s "$spgendir/$base.sp_warp" ) {
         shell( "$SOPR -d 2 -EXP $gendir/$base.sp_warp | $MCEP -a " . ( -$fw ) . " -m " . ( $fp / 2 ) . " -l $fp -q 3 -i 0 -j 0 | $MGC2SP -a 0.0 -m " . ( $fp / 2 ) . " -l $fp -o 2 > $gendir/$base.sp" );
      }

      if ( -s "$f0gendir/$base.lf0_delta" && -s "$f0gendir/$base.vuv" ) {
      	 if ($useMLPG){
			 $T = get_file_size("$f0gendir/$base.lf0_delta") / (4*$ordr{'lf0'}*$nwin{'lf0'});
			 shell("rm -f $gendir/$base.lf0_delta.var");
			 for ( $i = 1 ; $i <= $T ; $i++ ) {
				shell("cat $nndatdir/var/iplf0.var >> $gendir/$base.lf0_delta.var");
			 }
			 $dim = $ordr{'lf0'} * $nwin{'lf0'};
			 shell("$MERGE -s 0 -l $dim -L $dim $f0gendir/$base.lf0_delta < $gendir/$base.lf0_delta.var > $gendir/$base.lf0_pdf");
			 shell("$MLPG -l $ordr{'lf0'} -d $nndatdir/win/lf0.win2.f -d $nndatdir/win/lf0.win3.f $gendir/$base.lf0_pdf > $gendir/$base.lf0_ip");
         }else{
         	shell("mv $gendir/$base.lf0_delta $gendir/$base.lf0_ip");
         }
         shell("$F0VUV $gendir/$base.lf0_ip $f0gendir/$base.vuv > $gendir/$base.lf0");

         $f0gendir = "$gendir";
      }
      if ($synWav && -s "$f0gendir/$base.lf0" ) {
         lf02f0( "$f0gendir/$base.lf0", "$gendir/$base.f0" );
      }
      
      if ( -s "$apgendir/$base.bap_delta" ) {
      	 if ($useMLPG){
			 $T = get_file_size("$apgendir/$base.bap_delta") / (4*$ordr{'bap'}*$nwin{'bap'});
			 shell("rm -f $gendir/$base.bap_delta.var");
			 for ( $i = 1 ; $i <= $T ; $i++ ) {
				shell("cat $nndatdir/var/bap.var >> $gendir/$base.bap_delta.var");
			 }
			 $dim = $ordr{'bap'} * $nwin{'bap'};
			 shell("$MERGE -s 0 -l $dim -L $dim $apgendir/$base.bap_delta < $gendir/$base.bap_delta.var > $gendir/$base.bap_pdf");
			 shell("$MLPG -l $ordr{'bap'} -d $nndatdir/win/bap.win2.f -d $nndatdir/win/bap.win3.f $gendir/$base.bap_pdf > $gendir/$base.bap");
		 }else{
		 	 shell("mv $gendir/$base.bap_delta $gendir/$base.bap");
		 }
         $apgendir = "$gendir";
      }
      if ($synWav && -s "$apgendir/$base.bap") {
         bap2ap( "$gendir/$base.bap", "$gendir/$base.ap" );
      }
      
      if ($synWav && -s "$gendir/$base.sp" && -s "$gendir/$base.f0" && -s "$gendir/$base.ap" ) {
         shell("$X2X +fa $gendir/$base.f0 > $gendir/$base.f0.a");
         $line = "$SYNTHESIS_FFT -f $sr -spec -fftl $fp -shift " . ( ( $fs * 1000 ) / $sr ) . " -sigp 1.2 -cornf 4000 -float ";
         $line .= "-apfile $gendir/$base.ap $gendir/${base}.f0.a $gendir/$base.sp $gendir/$base.wav";
         shell($line);
         shell("rm -f $gendir/$base.f0.a");
      }

      shell("rm -f $gendir/$base.sp");
      shell("rm -f $gendir/$base.p_mgc $gendir/$base.mgc_delta.var $gendir/$base.mgc_pdf");
      shell("rm -f $gendir/$base.f0");
      shell("rm -f $gendir/$base.lf0_ip $gendir/$base.lf0_delta.var $gendir/$base.lf0_pdf");
      shell("rm -f $gendir/$base.ap");
      shell("rm -f $gendir/$base.bap_delta.var $gendir/$base.bap_pdf");
   }
}


sub gen_wave_only($$$$$$) {
   my ( $gendir, $spgendir, $f0gendir, $apgendir, $scp, $ext, $useSigPF2, $useMLPG, $synWav) = @_;
   my ( $tspgendir, $tf0gendir, $tapgendir);
   my ( $line, @FILE, $file, $base, $T, $dim );

   $tspgendir = $spgendir;
   $tf0gendir = $f0gendir;
   $tapgendir = $apgendir;

   @FILE = split( '\n', `cat $scp` );
   $lgopt = "-l" if ($lg);

   print "Processing directory $gendir:\n";

   foreach $file (@FILE) {
      $spgendir = $tspgendir;
      $f0gendir = $tf0gendir;
      $apgendir = $tapgendir;

      $base = `basename $file $ext`;
      chomp($base);
      
      if ($synWav && -s "$spgendir/$base.mgc" ) {
         if ( $useSigPF && $useSigPF2 ) {
            shell( "$MCPF -a $fw -m " . ( $ordr{'mgc'} - 1 ) . " -b $pf_mcp -l $il $spgendir/$base.mgc > $gendir/$base.p_mgc" );
            $mgc = "$gendir/$base.p_mgc";
         }
         else {
            $mgc = "$spgendir/$base.mgc";
         }

         if ( $gm == 0 ) {
            shell( "$MGC2SP -a $fw -g $gm -m " . ( $ordr{'mgc'} - 1 ) . " -l $fp -o 2 $mgc > $gendir/$base.sp" );
         }
         else {
            $line = "$LSPCHECK -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " -c -r 0.1 $mgc | ";
            $line .= "$LSP2LPC -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " $lgopt | ";
            $line .= "$MGC2MGC -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw -c $gm -n -u -M " . ( $ordr{'mgc'} - 1 ) . " -A $fw -C $gm | ";
            $line .= "$MGC2SP -a $fw -c $gm -m " . ( $ordr{'mgc'} - 1 ) . " -l $fp -o 2 > $gendir/$base.sp";
            shell($line);
         }         
      }

      if ($synWav && -s "$f0gendir/$base.lf0" ) {
         lf02f0( "$f0gendir/$base.lf0", "$gendir/$base.f0" );
      }
      
      if ($synWav && -s "$apgendir/$base.bap") {
         bap2ap( "$gendir/$base.bap", "$gendir/$base.ap" );
      }
      
      if ($synWav && -s "$gendir/$base.sp" && -s "$gendir/$base.f0" && -s "$gendir/$base.ap" ) {
         shell("$X2X +fa $gendir/$base.f0 > $gendir/$base.f0.a");
         $line = "$SYNTHESIS_FFT -f $sr -spec -fftl $fp -shift " . ( ( $fs * 1000 ) / $sr ) . " -sigp 1.2 -cornf 4000 -float ";
         $line .= "-apfile $gendir/$base.ap $gendir/${base}.f0.a $gendir/$base.sp $gendir/$base.wav";
         shell($line);
         shell("rm -f $gendir/$base.f0.a");
      }

      shell("rm -f $gendir/$base.sp");
      shell("rm -f $gendir/$base.p_mgc $gendir/$base.mgc_delta.var $gendir/$base.mgc_pdf");
      shell("rm -f $gendir/$base.f0");
      shell("rm -f $gendir/$base.lf0_ip $gendir/$base.lf0_delta.var $gendir/$base.lf0_pdf");
      shell("rm -f $gendir/$base.ap");
      shell("rm -f $gendir/$base.bap_delta.var $gendir/$base.bap_pdf");
   }
}


# Excluding silence ==============================
sub make_data_exsil($$$$) {
   my ($ordr, $lab, $data, $exdata, $ext) = @_;
   my ($i, $find, $tfind, $str, $start, $end, $path);

   $path = path_wo_ext($exdata, $ext);

   $i = 0;
   $find = 1;
   open( F, "$lab" ) || die "Cannot open $!";
   while ( $str = <F> ) {
      chomp($str);
      @arr = split( / /, $str );
      $tfind = $find;
      $find = 0;
      for ( $j = 0 ; $j < @slnt ; $j++ ) {
         if ( $arr[2] =~ /-$slnt[$j]+/ ) { 
            $find = 1;
            last;
         }
      }
      if ($find == 0 && $tfind == 1) {
         $i++;
         shell("rm -f $path.$i$ext");
      }
      if ( $find == 0 ) {
         $start = int( $arr[0] * ( 1.0e-7 / ( $fs / $sr ) ) + 0.5 );
         $end   = int( $arr[1] * ( 1.0e-7 / ( $fs / $sr ) ) + 0.5 ) - 1;
         shell("$BCUT -s $start -e $end -l $ordr +f $data >> $path.$i$ext");
      }
   }
}

sub num_exsildata($) {
   my ($lab) = @_;
   my ($i, $find, $tfind, $str);

   $i = 0;
   $find = 1;
   open( F, "$lab" ) || die "Cannot open $!";
   while ( $str = <F> ) {
      chomp($str);
      @arr = split( / /, $str );
      $tfind = $find;
      $find = 0;
      for ( $j = 0 ; $j < @slnt ; $j++ ) {
         if ( $arr[2] =~ /-$slnt[$j]+/ ) { 
            $find = 1;
            last;
         }
      }
      if ($find == 0 && $tfind == 1) {
         $i++;
      }
   }

   return ($i);
}

sub cmp_sil($$$$$) {
   my ($ordr, $lab, $sildata, $data, $cmpdata) = @_;

   shell("rm -f $cmpdata");
   open( F, "$lab" ) || die "Cannot open $!";
   while ( $str = <F> ) {
      chomp($str);
      @arr = split( / /, $str );
      $find = 0;
      for ( $j = 0 ; $j < @slnt ; $j++ ) {
         if ( $arr[2] =~ /-$slnt[$j]+/ ) { 
            $find = 1;
            last;
         }
      }
      $start = int( $arr[0] * ( 1.0e-7 / ( $fs / $sr ) ) + 0.5);
      $end   = int( $arr[1] * ( 1.0e-7 / ( $fs / $sr ) ) + 0.5) - 1;
      if ( $find == 0 ) {
         shell("$BCUT -s $start -e $end -l $ordr +f $data >> $cmpdata");
      }
      else {
         shell("$BCUT -s $start -e $end -l $ordr +f $sildata >> $cmpdata");
      }
      
   }

   return($i);
}

1;
