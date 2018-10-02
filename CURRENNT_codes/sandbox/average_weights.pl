#!/usr/bin/perl

# very simple weight averaging script
# does not do much error checking, so use with caution

use strict;
use IO::File;
#use JSON qw( decode_json );
use JSON::XS;
use Data::Dumper;

if ($#ARGV < 1) {
    print "Usage: $0 <in_nets> <out_net>\n";
    exit 1;
}

my @in = @ARGV[0..$#ARGV-1];
my $out = $ARGV[$#ARGV];

my @in_fh;

my %aj;
my %ctr;
for my $i (0..$#in) {
    print "$in[$i]\n";
    open(F, $in[$i]) or die "$in[$i]: $!";
    my @l = <F>;
    close(F);
    my $json = join("", @l);
    my $dj = decode_json($json);
    if ($i == 0) {
        %aj = %$dj;
    }
    print ".\n";
    #print Dumper $dj; #->{'layers'};
    for my $wk (keys %{$dj->{'weights'}}) {
        for my $wtk (keys %{$dj->{'weights'}->{$wk}}) {
            my $wref = $dj->{'weights'}->{$wk}->{$wtk};
            for my $j (0..$#{$wref}) {
                # incremental mean calculation
                $aj{'weights'}{$wk}{$wtk}[$j] += ($wref->[$j] - $aj{'weights'}{$wk}{$wtk}[$j]) / ($i + 1);
                #$ctr{'weights'}{$wk}{$wtk}[$i]++;
            }
            #print "$wk --> $wtk: @$wref\n";
        }
    }
}

open(OUT, '>', $out) or die "$out: $!";
print OUT JSON::XS->new->utf8->pretty(1)->encode(\%aj);
#print Dumper %aj;
close(OUT);
