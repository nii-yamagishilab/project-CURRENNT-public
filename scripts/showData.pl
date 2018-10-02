#!/usr/bin/perl

use File::Basename;
require($ARGV[0]) || die "Can't find config.pm";


my $j = 0;
foreach $ddir (@datadir){
    if (-e "$ddir/all.scp"){
	print "Print data.nc for $ddir\n";
	$dbuf = $buffdir[$j];
	open(IN_STR, "$ddir/all.scp");
	while(<IN_STR>){
	    chop;
	    $scpfile = $_;
	    $name = basename($scpfile);
	    $name =~ s/all\.scp/data\.nc/g;
	    $data = "$dbuf/$name";
	    print "$data,";
	}
	close(IN_STR);
	print "\n----\n";
    }else{
	print "all.scp has not been generated\n";
    }
    $j += 1;
}
