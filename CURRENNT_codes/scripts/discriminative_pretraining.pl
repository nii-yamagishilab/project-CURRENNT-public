#!/usr/bin/perl

use strict;
use JSON::XS;
use Data::Dumper;

sub run_train
{
    our ($currennt, $net_config, $train_nc, $val_nc, $test_nc, $max_epochs);
    my ($in_net, $out_net, $log_file, $learning_rate) = @_;
    my $cmd = "$currennt --train_file $train_nc ";
    if ($val_nc) {
        $cmd .= "--val_file $val_nc ";
    }
    if ($test_nc) {
        $cmd .= "--test_file $test_nc ";
    }
    #$cmd .= "--network $in_net --autosave_best true --autosave_prefix $out_net_prefix $net_config";
    $cmd .= "--network $in_net --save_network $out_net --max_epochs $max_epochs --autosave false --autosave_best false";
    if ($learning_rate > 0) {
        $cmd .= " --learning_rate $learning_rate";
    }
    $cmd .= " $net_config";

    print "$cmd\n";
    system("echo \"$cmd\" > $log_file");
    my $rv = system("$cmd 2>&1 | tee -a $log_file");
    if ($rv) {
        print "ERROR: Check $log_file\n";
        exit $rv;
    }
}


if ($#ARGV < 5) {
    print STDERR "Usage: $0 <in_net> <net_config> <work_dir> <train_nc> <val_nc> <test_nc> [max_epochs] [initial_lr lr_decay_factor]\n";
    exit 1;
}

our $currennt = "/home/iaa1/workstud/wen/tum-mmk-tools/currennt/build_5.0_gedas/currennt";
(my $in_net, our $net_config, my $work_dir, our $train_nc, our $val_nc, our $test_nc, our $max_epochs, my $initial_lr, my $lr_decay_factor) = @ARGV;
if (!$max_epochs) { $max_epochs = 50; }
if (!$initial_lr) { $initial_lr = -1; } # use learning rate from config or default
if (!$lr_decay_factor) { $lr_decay_factor = 1; }

open(F, $in_net) or die "$in_net: $!";
my @l = <F>;
close(F);
my $json = join("", @l);
# this is supposed to have no weights, but dimensions etc. for all the layers
my $initial_net = decode_json($json);

# assume that we have 1 output layer
# TODO: implement for nets without linear output layer
# subtract:
# - input
# - linear output
# - post output
my $num_hidden_layers = $#{$initial_net->{'layers'}} - 2;
print "Found $num_hidden_layers hidden layers\n";

# Save hidden layer sizes and types to re-add such layers later
my @hidden_sizes;
my @hidden_types;
for my $h (1..$num_hidden_layers) {
    $hidden_sizes[$h] = $initial_net->{'layers'}->[$h]->{'size'} + 0;
    $hidden_types[$h] = $initial_net->{'layers'}->[$h]->{'type'};
}

# remove all the hidden layers
my $net_to_train = $initial_net;
splice(@{$net_to_train->{'layers'}}, 1, $num_hidden_layers);

# remove all weights
delete $net_to_train->{'weights'};

mkdir($work_dir);

my $learning_rate = $initial_lr;
for my $hidden_layer_to_train (1..$num_hidden_layers)
{
    my $out_jsn_file = "$work_dir/trained.$hidden_layer_to_train.jsn";

    if (! -f $out_jsn_file) { 

        my %layer = ( 'name' => "hidden_layer_$hidden_layer_to_train", "type" => $hidden_types[$hidden_layer_to_train], "size" => $hidden_sizes[$hidden_layer_to_train] + 0.0, "bias" => 1.0 ) ;
        splice(@{$net_to_train->{'layers'}}, $hidden_layer_to_train, 0, \%layer);

        # remove output layer weights
        # assume that the name of the output layer is "output".
        delete $net_to_train->{'weights'}->{'output'};
        my $jsn_file = "$work_dir/train.$hidden_layer_to_train.jsn";
        open(JSN, '>', $jsn_file) or die "$jsn_file: $!";
        print JSN JSON::XS->new->utf8->pretty(1)->encode($net_to_train);
        close(JSN);

        my $log_file = "$work_dir/pretrain.$hidden_layer_to_train.log";
        run_train($jsn_file, $out_jsn_file, $log_file, $learning_rate);

    }

    # read pretrained net with weights
    open(F, $out_jsn_file) or die "$out_jsn_file: $!";
    my @l = <F>;
    close(F);
    $net_to_train = decode_json(join('', @l));

    $learning_rate *= $lr_decay_factor;
}

# done
# result: <work_dir>/trained.<nlayers>.jsn
