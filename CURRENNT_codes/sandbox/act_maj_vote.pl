#!/usr/bin/perl
use strict;
my $in = $ARGV[0];
my $nclass = 41;
open(IN, $in) or die "$in: $!";
my $i = 0;
while (<IN>) {
    my @els = split(/;/);
    my @ctr;
    my @scores;
    if ($#els % $nclass != 0) {
        print "ERROR: Wrong number of classes!\n";
        exit 1;
    }
    for (my $i = 1; $i < $#els; $i += $nclass) {
        for (my $j = 0; $j < $nclass; ++$j) {
            $scores[$j] += $els[$i + $j];
        }
    }
    my $maxi = 0;
    my $max = $scores[0];
    #print "@scores\n";
    for (my $j = 0; $j < $nclass; ++$j) {
        if ($max < $scores[$j]) {
            $max = $scores[$j];
            $maxi = $j;
        }
    }
    print "$els[0] $maxi\n";
}
close(IN);
