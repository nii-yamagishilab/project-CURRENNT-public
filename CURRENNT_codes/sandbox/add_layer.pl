#!/usr/bin/perl

# very simple weight averaging script
# does not do much error checking, so use with caution

use strict;
use IO::File;
#use JSON qw( decode_json );
use JSON::XS;
use Data::Dumper;

if ($#ARGV < 4) {
    print "Usage: $0 <in_net> <out_net> <name> <type> <size>\n";
    exit 1;
}

my ($in, $out, $name, $type, $size) = @ARGV;

open(F, $in) or die "$in: $!";
my @l = <F>;
close(F);
my $json = join("", @l);
my $dj = decode_json($json);
my %aj = %$dj;
my $nl = $#{$dj->{'layers'}};
%{$aj{'layers'}[$nl+1]} = %{$dj->{'layers'}->[$nl]};
%{$aj{'layers'}[$nl]} = %{$dj->{'layers'}->[$nl-1]};
$aj{'layers'}[$nl-1]{'name'} = $name;
$aj{'layers'}[$nl-1]{'type'} = $type;
$aj{'layers'}[$nl-1]{'size'} = $size + 0;
$aj{'layers'}[$nl-1]{'bias'} = 1.0;
delete $aj{'weights'}{$dj->{'layers'}->[$nl]->{'name'}};

open(OUT, '>', $out) or die "$out: $!";
print OUT JSON::XS->new->utf8->pretty(1)->encode(\%aj);
#print Dumper %aj;
close(OUT);
