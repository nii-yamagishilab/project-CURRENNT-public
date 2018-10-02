#!/usr/bin/perl

if(@ARGV<1){
    die "Usage: perl 000_PRE.pl config.pm";
}else{
    require($ARGV[0]);
}

if (@datadir != @sysdir || @sysdir != @testdatadir){
    die "Make sure \@sysdir and \@datadir have same length";
}

my $i = 0;
print_time("Preparing");
foreach my $ddir (@datadir){
    $sdir = $sysdir[$i];
    $tdir = $testdatadir[$i];
    SelfSystem("mkdir -p $sdir");
    SelfSystem("mkdir -p $tdir");
    SelfSystem("mkdir -p $ddir");
    unless(-e $sdir){die "Can't create $sdir. Maybe parent directory should be created?";}
    unless(-e $tdir){die "Can't create $tdir. Maybe parent directory should be created?";}
    unless(-e $ddir){die "Can't create $ddir. Maybe parent directory should be created?";}
    unless(-e "$ddir/data_config.py"){
    $command = "cp ./template/data_config.py_train $ddir/data_config.py";
    SelfSystem($command);}
    unless(-e "$tdir/data_config.py"){
    $command = "cp ./template/data_config.py_test $tdir/data_config.py";
    SelfSystem($command);}
    unless(-e "$sdir/network.jsn"){
    $command = "cp ./template/network.jsn $sdir/network.jsn";
    SelfSystem($command);}
    unless(-e "$sdir/config.py"){
    $command = "cp ./template/config.cfg $sdir/config.cfg";
    SelfSystem($command);}
        
    $i += 1;
}



$i = 0;
print_time("NOTE");
print "Please modify the data_config.py in each folder\n";
print "Please modify the network.jsn and config.cfg in each folder\n";

foreach my $ddir (@datadir){

    $sdir = $sysdir[$i];
    $tdir = $testdatadir[$i];
    $command = "$ddir/data_config.py";
    print $command,"\n";
    $command = "$tdir/data_config.py";
    print $command,"\n";
    $command = "$sdir/network.jsn";
    print $command,"\n";
    $command = "$sdir/config.cfg";
    print $command,"\n";
    $i += 1;
}
print "Please prepare the test folder @testdatadir\n\t\tRemember to specify full mask.txt\n";
print "Please remember:\n\t\tSet PYTHONPATH\n\t\tSet CURRENNT_CUDA_DEVICE\n";
