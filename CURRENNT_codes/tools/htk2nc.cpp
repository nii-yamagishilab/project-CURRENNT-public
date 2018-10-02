#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "netcdf.h"


using namespace std;


void swap32 (uint32_t *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}


void swap16 (uint16_t *p) 
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 1 ); *( q + 1 ) = temp;
}


void swapFloat(float *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}


void swapFloatCopyArray(float *src, float *dst, size_t n)
{
    uint8_t *src_ = (uint8_t*) src;
    uint8_t *dst_ = (uint8_t*) dst;
    size_t n4 = n << 2;
//    cout << "n4 = " << n4 << endl;
    for (size_t i = 0; i < n4; i+= 4) {
        dst_[i]     = src_[i + 3];
        dst_[i + 1] = src_[i + 2];
        dst_[i + 2] = src_[i + 1];
        dst_[i + 3] = src_[i];
    }
}

// copy float matrix to sub-matrix of destination specified by column offset
// m: number of rows of both matrices
// nsrc: number of columns of source matrix
// ndst: number of columns of target matrix
// off: column offset where matrix is copied to
void swapFloatCopy2DArray(float *src, float *dst, size_t m, size_t nsrc, size_t ndst, ptrdiff_t off)
{
    uint8_t *src_ = (uint8_t*) src;
    uint8_t *dst_ = (uint8_t*) dst;
    uint8_t *p_src = src_;
    uint8_t *p_dst = dst_;
    uint8_t *p_dst_row = dst_;
    for (size_t i = 0; i < m; ++i) {
        p_dst = p_dst_row + (off << 2);
        for (size_t j = 0; j < nsrc; ++j) {
        //cout << "(" << i << "," << j << ")" << endl;
            // copy and swap single element
            *p_dst       = *(p_src + 3);
            *(p_dst + 1) = *(p_src + 2);
            *(p_dst + 2) = *(p_src + 1);
            *(p_dst + 3) = *p_src;
            p_src += 4;
            p_dst += 4;
        }
        p_dst_row += (ndst << 2);
    }
}

struct htkdata {
    uint32_t nSamples;
    uint32_t samplePeriod;
    uint16_t sampleSize;
    uint16_t sampleKind;
    float* rawData; // NOT swapped
};

int readHtk(const char* filename, htkdata* dst, bool headerOnly = false)
{
    ifstream htkstream(filename);
    if (htkstream.good()) {
        htkstream.read((char*)(&dst->nSamples), sizeof(uint32_t));
        if (htkstream.gcount() != sizeof(uint32_t))
        {
            return -1;
        }
        swap32(&dst->nSamples);
        //cout << dst->nSamples << endl;
    }
    else {
        return -1;
    }
    if (htkstream.good()) {
        htkstream.read((char*)(&dst->samplePeriod), sizeof(uint32_t));
        if (htkstream.gcount() != sizeof(uint32_t))
        {
            return -1;
        }
        swap32(&dst->samplePeriod);
        //cout << dst->samplePeriod << endl;
    }
    if (htkstream.good()) {
        htkstream.read((char*)(&dst->sampleSize), sizeof(uint16_t));
        if (htkstream.gcount() != sizeof(uint16_t))
        {
            return -1;
        }
        swap16(&dst->sampleSize);
        //cout << dst->sampleSize << endl;
    }
    if (htkstream.good()) {
        htkstream.read((char*)(&dst->sampleKind), sizeof(uint16_t));
        if (htkstream.gcount() != sizeof(uint16_t))
        {
            return -1;
        }
        swap16(&dst->sampleKind);
        //cout << dst->sampleKind << endl;
    }

    if (headerOnly)
        return 0;

    uint16_t nComps = dst->sampleSize / sizeof(float); // assuming float HTK format here ... but int should be the same ...
    if (htkstream.good()) {
        dst->rawData = new float[nComps * dst->nSamples];
        size_t data_size = dst->sampleSize * dst->nSamples;
        htkstream.read((char*)dst->rawData, data_size);
        /*float test[1];
        swapFloatCopyArray(dst->rawData, test, 1);
        cout << "float test. " << test[0] << endl;*/
        if (htkstream.gcount() != data_size) {
            return -1;
        }
        //cout << "read " << data_size << " bytes " << endl;
    }
    return 0;
}


// read label file and setup label map, get number of timesteps
int read_label_file(const char* filename, map<string, int>* labelMap, uint32_t* len)
{
    ifstream lfstream(filename);
    if (lfstream.good()) {
        string buf;
        while (!lfstream.eof()) {
            getline(lfstream, buf);
//            cout << buf << endl;
            if (!buf.empty()) {
                if (labelMap->find(buf) == labelMap->end()) 
                    labelMap->insert(pair<string, int>(buf, 1));
                ++(*len);
            }
        }
    }
    else {
        return -1;
    }
    int labelIdx = 0;
    for (map<string, int>::iterator itr = labelMap->begin(); itr != labelMap->end(); ++itr)
    {
        itr->second = labelIdx++;
    }
    return 0;
}


// read label file and get label indices as int
int read_label_file(const char* filename, const map<string, int>& labelMap, int* label_buf, size_t nlabels)
{
    ifstream lfstream(filename);
    size_t j = 0;
    if (lfstream.good()) {
        string buf;
        while (!lfstream.eof()) {
            getline(lfstream, buf);
            if (!buf.empty()) {
                map<string, int>::const_iterator item = labelMap.find(buf);
                if (item == labelMap.end()) {
                    return -1;
                }
                else {
                    if (j > nlabels) {
                        return -1;
                    }
                    label_buf[j] = item->second;
                }
                ++j;
            }
        }
    }
    else {
        return -1;
    }
    return 0;
}


int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <mapping.txt> <out.nc> [map file delimiter character] [max. sequence length, 0 for infinite]" << endl;
        cerr << "  where mapping.txt defines mappings of HTK files" << endl;
        cerr << "  (multiple targets will be combined)" << endl;
        cerr << "Mapping syntax:" << endl;
        cerr << "  <seq_tag> <#input files> <input_feat_file> [ <input_feat_file> ... ] <output_feat_file> [ <output_feat_file> ... ]" << endl;
        cerr << "Ex." << endl;
        cerr << "  seq1 1 seq1_noisy.htk seq1_speech.htk seq1_noise.htk" << endl;
        cerr << "  seq2 1 seq2_noisy.htk seq2_speech.htk seq2_noise.htk" << endl;
        cerr << "  ..." << endl;
        return 1;
    }
    
    ifstream fs(argv[1]);
    string buf, tok;
    uint32_t totalTimesteps = 0;
    uint32_t input_size = 0;
    int nInputs = 0;
    vector<uint32_t> vect_sizes;
    uint32_t output_size = 0;
    vector<vector<string> > mapping;
    vector<int> seqLens;
    int max_fn_len = 0;
    vector<string> seqTags;
    vector<vector<string> > labels; // let's keep multi-task classification in mind ...
    int maxLabelLength = 0;
    int nClassificationTasks = 0;
    vector<map<string, int> > labelMap; 

    char mappingDelim = ' ';
    int maxSeqLen = 0;
    if (argc > 3) {
      mappingDelim = argv[3][0];
    }
    if (argc > 4) {
      maxSeqLen = atoi(argv[4]);
      cout << "Max sequence length is " << maxSeqLen << endl;
    }
    bool isClassification = false;

    bool first = true;
    
    // parse mapping file
    while (!fs.eof()) {
        getline(fs, buf);
        if (buf.empty())
            break;
        stringstream ss(buf);
        vector<string> tokens;
        while (ss) {
          string s;
          if (!getline( ss, s, mappingDelim )) 
            break;
          if (s.length() > 0) 
            tokens.push_back(s);
        }
/*        while (ss >> tok) {
            tokens.push_back(tok);
        }
*/
        if (tokens.size() > 1) {
            seqTags.push_back(tokens[0]);
            if (tokens[0].size() + 1 > max_fn_len) {
                max_fn_len = tokens[0].size() + 1;
            }
            // remove sequence tag from token list
            tokens.erase(tokens.begin(), tokens.begin() + 1);
	    // read number of input htk files to concat
            int nInputsLocal = atoi(tokens[0].c_str());
	    if (nInputsLocal <= 0 || nInputsLocal >= tokens.size() - 1) {
              cerr << "Number of input HTK files (2nd column) is out of range!" << endl;
              return -1;
            }
            if (first) {
              nInputs = nInputsLocal;
            } else if (nInputs != nInputsLocal) {
              cerr << "Inconsistent number of input htk files! Must be the same for all." << endl;
              return -1;
            }
            tokens.erase(tokens.begin(), tokens.begin() + 1);
            // TODO: if 2nd col is no number, assume 1 input file in 2nd col.
            //   if 2nd col is missing on non-first line, use first value
            // verify and count dimensions
            uint32_t seqLen = 0;
            if (first) {
                vect_sizes.resize(tokens.size(), 0);
            }
            else if (vect_sizes.size() != tokens.size()) {
                cerr << "Expected " << vect_sizes.size() << " filenames!" << endl;
                return -1;
            }
            for (int f = 0; f < tokens.size(); ++f) {
                uint32_t thisSeqLen = 0;
                // XXX: somewhat "magic" switch for classification ...
                if ((tokens[f].size() >= 4 && tokens[f].substr(tokens[f].size() - 4, 4) == ".txt")
                    || (tokens[f].size() >= 7 && tokens[f].substr(tokens[f].size() - 7, 7) == ".labels"))
                {
                    if (f == 0) {
                        cerr << "Input file must not be in text format!" << endl;
                        return -1;
                    }
                    if (tokens.size() > 2) {
                        cerr << "Multi-task classification currently unsupported!" << endl;
                        return -1;
                    }
                    isClassification = true;
                    if (first) {
                        ++nClassificationTasks;
                        vect_sizes[f] = 1; // needed?!
                        labelMap.resize(nClassificationTasks);
                    }
                    // read text file, compute # samples (seq len)
                    int ret = read_label_file(tokens[f].c_str(), &(labelMap[nClassificationTasks - 1]), &thisSeqLen);
                    //cout << "read " << thisSeqLen << " labels from file " << tokens[f] << endl;
                    if (ret != 0) {
                        cerr << "Could not read label file: " << tokens[f] << endl;
                        return ret;
                    }
                } // text (labels)
                else {
                    htkdata htmp;
                    int ret = readHtk(tokens[f].c_str(), &htmp, true);
                    if (ret != 0) {
                        cerr << "Could not read htk data from file " << tokens[f] << endl;
                        return ret;
                    }
                    int nComps = htmp.sampleSize / sizeof(float);
                    if (first) {
                        vect_sizes[f] = nComps;
                        if (f >= nInputs) {
                            output_size += nComps;
                        } else {
                            input_size += nComps;
                        }
                    }
                    else if (vect_sizes[f] != nComps) {
                        cerr << "Vector size mismatch: " << nComps << " vs. " << vect_sizes[f] << endl;
                    }
                    thisSeqLen = htmp.nSamples;
                } // HTK
                
                if (f > 0) {
                    if (thisSeqLen != seqLen) {
                        cerr << "WARNING: sequence length mismatch in files: " << thisSeqLen << " vs. " << seqLen << endl;
                        //return -1;
                        if (thisSeqLen < seqLen) {
                            seqLen = thisSeqLen;
                        }
                        cerr << " - setting length[" << seqLens.size() + 1 << "] to " << seqLen << endl;
                    }
                }
                else {
                    seqLen = thisSeqLen;
                }
            }
/*            if (first) {
                input_size = vect_sizes[0];
            }*/
            totalTimesteps += seqLen;
            mapping.push_back(tokens);
            seqLens.push_back(seqLen);
        }
        else {
            cerr << "Error: expected at least 2 filenames in file " << argv[1] << endl;
            return -1;
        }
        first = false;
    }
    cout << "Total timesteps: " << totalTimesteps << endl;
    int nSeq = mapping.size();
    cout << "# of sequences: " << nSeq << endl;
    cout << "input size: " << input_size << endl;
    
    if (isClassification) {
        int task = 0;
        labels.resize(nClassificationTasks);
        for (vector<map<string, int> >::const_iterator itr = labelMap.begin(); itr != labelMap.end(); ++itr)
        {
            ++task;
            cout << "Classification task #" << task << ": " << itr->size() << " labels" << endl;
            for (map<string, int>::const_iterator itr2 = itr->begin(); itr2 != itr->end(); ++itr2) 
            {
                cout << "  " << itr2->second << ": " << itr2->first << endl;
                labels[task - 1].push_back(itr2->first);
                if (itr2->first.size() + 1 > maxLabelLength) {
                    maxLabelLength = itr2->first.size() + 1;
                }
            }
        }
    }
    else {
        cout << "output size: " << output_size << endl;
    }

    // hack to limit sequence length
    float tolPercent = 0.05;
    int nNewSeq = 0;
    if (maxSeqLen == 0) {
      nNewSeq = nSeq;
    } else {
      for (int s = 0; s < nSeq; ++s) {
        float d = (float)seqLens[s] / (float)maxSeqLen;
        d -= tolPercent;
        if (d < 1.0/(float)maxSeqLen) { d = 1.0/(float)maxSeqLen; }
        nNewSeq += (int)ceil(d);
//cout << "new seq " << nNewSeq << " and d = " << d << endl;
      }
    }

    // convert seq len vector to C array
    int seqLenArr[nNewSeq];
    if (maxSeqLen == 0) {
      for (int s = 0; s < nSeq; ++s) {
        seqLenArr[s] = seqLens[s];
      }
    } else {
      vector<string> seqTagsNew;
      int l = 0;
      for (int s = 0; s < nSeq; ++s) {
        float d = (float)seqLens[s] / (float)maxSeqLen - tolPercent;
        if (d < 1.0/(float)maxSeqLen) { d = 1.0/(float)maxSeqLen; }
        int di = ceil(d);
        int tmplen = seqLens[s];
        for (int i = 0; i < di - 1; i++) {
//cout << "seqLenArr["<<l<<"] = "<<maxSeqLen<<endl;
          seqLenArr[l++] = maxSeqLen;
          char tmpc[30];
          sprintf(tmpc, "%i", i+1);
          string tmp(tmpc);
          string tmp2(seqTags[s] + "--" + tmp);
          if (tmp2.size() + 1 > max_fn_len) {
            max_fn_len = tmp2.size() + 1;
          }
          seqTagsNew.push_back(tmp2);
          tmplen -= maxSeqLen;
        }
//cout << "seqLenArr["<<l<<"] = "<<tmplen<<endl;
        seqLenArr[l++] = tmplen;
        char tmpc[30];
        sprintf(tmpc, "%i", di);
        string tmp(tmpc);
        string tmp2(seqTags[s] + "--" + tmp);
        if (tmp2.size() + 1 > max_fn_len) {
          max_fn_len = tmp2.size() + 1;
        }
        seqTagsNew.push_back(tmp2);
        //seqTagsNew.push_back(seqTags[s]);
      }
      seqTags = seqTagsNew;
      // TODO: update seqTags array!
    }

    int numLabels = 0;
    if (isClassification)
        numLabels = labelMap[0].size(); // for NC generation only 1 task is supported ...

    // create nc "header"

    // dimensions
    int ret;
    int ncid;
    //if ((ret = nc_create(argv[2], NC_64BIT_OFFSET, &ncid)) != NC_NOERR) {
    if ((ret = nc_create(argv[2], NC_NETCDF4, &ncid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }

    int nseq_dimid;
    if ((ret = nc_def_dim(ncid, "numSeqs", nNewSeq, &nseq_dimid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }
    int nt_dimid;
    if ((ret = nc_def_dim(ncid, "numTimesteps", totalTimesteps, &nt_dimid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }
    int input_size_dimid;
    if ((ret = nc_def_dim(ncid, "inputPattSize", input_size, &input_size_dimid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }
    int output_size_dimid;
    int num_labels_dimid;
    if (isClassification) {
        if ((ret = nc_def_dim(ncid, "numLabels", numLabels, &num_labels_dimid)) != NC_NOERR) {
            cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
            return ret;
        }
    }
    else {
        if ((ret = nc_def_dim(ncid, "targetPattSize", output_size, &output_size_dimid)) != NC_NOERR) {
            cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
            return ret;
        }
    }
    int mstl_dimid;
    if ((ret = nc_def_dim(ncid, "maxSeqTagLength", max_fn_len, &mstl_dimid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }

    int mll_dimid;
    if (isClassification) {
        if ((ret = nc_def_dim(ncid, "maxLabelLength", maxLabelLength, &mll_dimid)) != NC_NOERR) {
            cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
            return ret;
        }
    }
    
    // VARIABLE DEFINITIONS:
    
    // labels
    int class_list_varid;
    if (isClassification) {
        int class_list_dimids[] = {num_labels_dimid, mll_dimid};
        if ((ret = nc_def_var(ncid, "labels", NC_CHAR, 2, class_list_dimids, &class_list_varid)) != NC_NOERR) {
            cerr << "Could not create NC file (labels): " << nc_strerror(ret) << endl;
            return ret;
        }
    }
    
    // seqTags
    int seqTags_dimids[] = {nseq_dimid, mstl_dimid};
    int seqTags_varid;
    if ((ret = nc_def_var(ncid, "seqTags", NC_CHAR, 2, seqTags_dimids, &seqTags_varid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }

    // seqLengths
    int seqLengths_dimids[] = {nseq_dimid};
    int seqLengths_varid;
    if ((ret = nc_def_var(ncid, "seqLengths", NC_INT, 1, seqLengths_dimids, &seqLengths_varid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }

    // inputs
    int input_dimids[] = { nt_dimid, input_size_dimid };
    int inputs_varid;
    if ((ret = nc_def_var(ncid, "inputs", NC_FLOAT, 2, input_dimids, &inputs_varid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }

    int labels_varid;
    int outputs_varid;
    if (isClassification) {
        // targetClasses
        int labels_dimids[] = { nt_dimid };
        if ((ret = nc_def_var(ncid, "targetClasses", NC_INT, 1, labels_dimids, &labels_varid)) != NC_NOERR) {
            cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
            return ret;
        }
    }
    else {
        // targetPatterns
        int output_dimids[] = { nt_dimid, output_size_dimid };
        if ((ret = nc_def_var(ncid, "targetPatterns", NC_FLOAT, 2, output_dimids, &outputs_varid)) != NC_NOERR) {
            cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
            return ret;
        }
    }

    // exit definition mode to write variable contents.
    if ((ret = nc_enddef(ncid)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }

    // "labels" (class list!)
    if (isClassification) {
        for (int l = 0; l < numLabels; ++l) {
            size_t start[] = {l, 0};
            size_t count[] = {1, labels[0][l].size()};
            //cout << "put " << labels[0][l] << endl;
            if ((ret = nc_put_vara_text(ncid, class_list_varid, start, count, labels[0][l].c_str())) != NC_NOERR) {
                cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
                return ret;
            }
        }
    }

    // seqTags
    for (int s = 0; s < nNewSeq; ++s) {
        size_t start[] = {s, 0};
        size_t count[] = {1, seqTags[s].size()};
        if ((ret = nc_put_vara_text(ncid, seqTags_varid, start, count, seqTags[s].c_str())) != NC_NOERR) {
            cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
            return ret;
        }
    }
    
    // seqLengths
    if ((ret = nc_put_var_int(ncid, seqLengths_varid, seqLenArr)) != NC_NOERR) {
        cerr << "Could not create NC file: " << nc_strerror(ret) << endl;
        return ret;
    }
    
    // inputs, targetPatterns
    size_t t = 0;
    for (int s = 0; s < nSeq; ++s) {
        if (s > 0 && s % 100 == 0)
            cout << s << endl;
        float* inp_buf = new float[input_size * seqLens[s]];
        int* label_buf; //= new int[
        float* outp_buf;
        if (isClassification) {
            label_buf = new int[seqLens[s]];
        }
        else {
            outp_buf = new float[output_size * seqLens[s]];
        }


        htkdata h;
        int input_start = 0;
        // concatenate into single buffer
        for (int f = 0; (f < mapping[s].size()) && (f < nInputs); ++f) {
//cerr << "f = "<<f<<" s= "<<s<<endl;
            ret = readHtk(mapping[s][f].c_str(), &h);
            if (ret != 0) {
                cerr << "Could not read htk data from file " << mapping[s][f] << endl;
                return ret;
            }
            //cout << "target file: " << h.nSamples << " x " << h.sampleSize / 4 << endl;
            // (seqLen x vect_size) matrix to (seqLen x output_size) matrix at offset output_start
            swapFloatCopy2DArray(h.rawData, inp_buf, seqLens[s], vect_sizes[f], input_size, input_start);
            delete[] h.rawData;
            input_start += vect_sizes[f];
        }
        // write buffer
        size_t start[] = {t, 0};
        size_t count[] = {seqLens[s], input_size};
        if ((ret = nc_put_vara_float(ncid, inputs_varid, start, count, inp_buf)) != NC_NOERR) {
             cerr << "Could not write inputs: " << nc_strerror(ret) << endl;
             return ret;
        }
        delete[] inp_buf;

        //cout << "input file: " << h.nSamples << " x " << h.sampleSize / 4 << endl;
//        if ((ret = nc_put_vara_float(ncid, inputs_varid, start, count, inp_buf)) != NC_NOERR) {
//            cerr << "Could not write inputs: " << nc_strerror(ret) << endl;
//            return ret;
//        }
        // labels
        if (isClassification) {
            // assume 1 task ...
            ret = read_label_file(mapping[s][nInputs].c_str(), labelMap[0], label_buf, seqLens[s]);
            if (ret != 0) {
                cerr << "Error reading label file file " << mapping[s][nInputs] << endl;
                return ret;
            }
            size_t start_l[] = {t};
            size_t count_l[] = {seqLens[s]};
            if ((ret = nc_put_vara_int(ncid, labels_varid, start_l, count_l, label_buf)) != NC_NOERR) {
                cerr << "Could not write target labels: " << nc_strerror(ret) << endl;
                return ret;
            }
            delete[] label_buf;
        }
        // regression outputs
        else {
            int output_start = 0;
            // concatenate into single buffer
            for (int f = nInputs; f < mapping[s].size(); ++f) {
                ret = readHtk(mapping[s][f].c_str(), &h);
                if (ret != 0) {
                    cerr << "Could not read htk data from file " << mapping[s][f] << endl;
                    return ret;
                }
                //cout << "target file: " << h.nSamples << " x " << h.sampleSize / 4 << endl;
                // (seqLen x vect_size) matrix to (seqLen x output_size) matrix at offset output_start
                swapFloatCopy2DArray(h.rawData, outp_buf, seqLens[s], vect_sizes[f], output_size, output_start);
                delete[] h.rawData;
                output_start += vect_sizes[f];
            }
            // write buffer
            count[1] = output_size;
            if ((ret = nc_put_vara_float(ncid, outputs_varid, start, count, outp_buf)) != NC_NOERR) {
                cerr << "Could not write outputs: " << nc_strerror(ret) << endl;
                return ret;
            }
            delete[] outp_buf;
        }
        //delete[] inp_buf;
        t += seqLens[s];
    }


    nc_close(ncid);
    return 0;
}
