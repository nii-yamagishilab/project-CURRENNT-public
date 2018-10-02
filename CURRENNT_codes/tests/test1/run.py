#!/usr/bin/python
import subprocess;
import json;

maxWeightDiff = 1e-10

subprocess.call(['../../build/currennt', 'config.cfg'])

actual   = json.load(open('trained_network.jsn'))
expected = json.load(open('expected_network.jsn'))

# check section 'layers'
if actual['layers'] != expected['layers']:
	print('The layers sections differ!')
	exit(1)

# check section 'weights'
for layer in actual['weights']:
	for type in actual['weights'][layer]:
		actualWeights   = actual  ['weights'][layer][type]
		expectedWeights = expected['weights'][layer][type]
		for i in range(len(actualWeights)):
			if abs(actualWeights[i] - expectedWeights[i]) > maxWeightDiff:
				print('Different weights in weights.{}.{}[{}]:'.format(layer, type, i))
				print('Expected: {}'.format(expectedWeights[i]))
				print('Actual:   {}'.format(actualWeights[i]))
				exit(1)

print('Test successful')
exit(0)

