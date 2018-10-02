// nc-standardize: standardize inputs and maybe output features in nc file
// used for network training etc.

// compile with: g++ nc-standardize.cpp -lnetcdf -onc-standardize

#include "netcdf.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>


using namespace std;


int createVarIfNotExists(int ncid, const char* varName, const char* dimName, int* varid, float* data)
{
    int ret;

    // try to retrieve specified dimension
    int dimid;
    size_t N;
    if ((ret = nc_inq_dimid(ncid, dimName, &dimid)) != NC_NOERR) {
        return ret;
    }
    if ((ret = nc_inq_dimlen(ncid, dimid, &N)) != NC_NOERR) {
        return ret;
    }

    size_t start[] = {0};
    size_t count[] = {N};

    // try to retrieve specified variable
    ret = nc_inq_varid(ncid, varName, varid);
    
    if (ret != NC_NOERR) {
        // variable not found, try to create it
        // enter definition mode
        if ((ret = nc_redef(ncid)) != NC_NOERR) {
            return ret;  
        }          
        // define variable
        if ((ret = nc_def_var(ncid, varName, NC_FLOAT, 1, &dimid, varid)) != NC_NOERR) {
            return ret;
        }
        // exit definition mode
        if ((ret = nc_enddef(ncid)) != NC_NOERR) {
            return ret;
        }
    }
    
    // all good, write data
    if ((ret = nc_put_vara_float(ncid, *varid, start, count, data)) != NC_NOERR) {
        return ret;
    }
    
    return 0;
}


// XXX: we could make these functions being exported by currennt_lib


int readNcDimension(int ncid, const char *dimName)
{
    int ret;
    int dimid;
    size_t x;

    if ((ret = nc_inq_dimid(ncid, dimName, &dimid)) || (ret = nc_inq_dimlen(ncid, dimid, &x)))
        throw std::runtime_error(std::string("Cannot get dimension '") + dimName + "': " + nc_strerror(ret));

    return (int)x;
}


int readNcIntArray(int ncid, const char *arrName, int arrIdx)
{
    int ret;
    int varid;
    size_t start[] = {arrIdx};
    size_t count[] = {1};

    int x;
    if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_int(ncid, varid, start, count, &x)))
        throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

    return x;
}


float* readNcFloatArray(int ncid, const char *arrName, float* ptr, size_t n)
{
    int ret;
    int varid;
    size_t start[] = {0};
    size_t count[] = {n};

    if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_float(ncid, varid, start, count, ptr)))
        throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

    return ptr;
}


float* readNcPatternArray(int ncid, const char *arrName, int begin, int n, int patternSize, int* save_varid = 0)
{
    int ret;
    int varid;
    size_t start[] = {begin, 0};
    size_t count[] = {n, patternSize};

    float* v = new float[n * patternSize];
    if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_float(ncid, varid, start, count, v)))
        throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

    if (save_varid != 0)
        *save_varid = varid;

    return v;
}



int main(int argc, char** argv)
{
    // compute means / variances
    // OR load normdata
    
    // nc-standardize train.nc -        # save parms into train.nc
    // nc-standardize dev.nc train.nc  # load parms from train.nc
    // nc-standardize test.nc train.nc # load parms from train.nc
    
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <nc-file> [normdata [-save]]" << endl;
        return 1;
    }
    
    char* ncfile = argv[1];
    
    int ncid;
    int ret;
    
    bool std_output = true;
    if (std::strlen(argv[0]) >= 20 && !std::strncmp(argv[0] + std::strlen(argv[0]) - 6, "-input", 6)) {
        std_output = false;
        cout << argv[0] << ": do not standardize output features" << std::endl;
    }
    
    if ((ret = nc_open(ncfile, NC_WRITE, &ncid)))
        throw std::runtime_error(std::string("Could not open '") + ncfile + "': " + nc_strerror(ret));

    int input_size = readNcDimension(ncid, "inputPattSize");
    cout << "Input size: " << input_size << endl;
    int output_size = 1;
    try {
        output_size = readNcDimension(ncid, "targetPattSize");
        cout << "Output size: " << output_size << endl;
    }
    catch (...) {
        std_output = false;
        cerr << "WARNING: targetPattSize field not found, do not standardize outputs (classification task?)" << endl;
    }
    
    int total_sequences = readNcDimension(ncid, "numSeqs");
    cout << "# of sequences: " << total_sequences << endl;
    
    bool compute_normdata = strlen(argv[2]) == 1 && argv[2][0] == '-'; 
    
    double input_means_tmp[input_size];
    double input_sqmeans_tmp[input_size];
    
    double output_means_tmp[output_size];
    double output_sqmeans_tmp[output_size];
    
    float input_means[input_size];
    float input_sds[input_size];
    
    float output_means[output_size];
    float output_sds[output_size];
    
    if (compute_normdata) {
        for (int i = 0; i < input_size; ++i) {
            input_means[i] = 0.0f;
            input_sds[i] = 0.0f;        
            input_means_tmp[i] = 0.0f;
            input_sqmeans_tmp[i] = 0.0f;
        }
        for (int i = 0; i < output_size; ++i) {
            output_means[i] = 0.0f;
            output_sds[i] = 0.0f;        
            output_means_tmp[i] = 0.0f;
            output_sqmeans_tmp[i] = 0.0f;
        }
    
        // compute means/variances
        int start = 0;
        int total_timesteps = 0;
        for (int i = 0; i < total_sequences; ++i) {
            int seqLength = readNcIntArray(ncid, "seqLengths", i);
            float* inputs = readNcPatternArray(ncid, "inputs", start, seqLength, input_size);
            //cout << "seq #" << i << " length = " << seqLength << endl;
            for (int t = 0; t < seqLength; ++t) {
                for (int j = 0; j < input_size; ++j) {
                    float tmp = inputs[t * input_size + j];
                    /* Use rapid calculation according to Welford, 1962 */
                    double k = t + total_timesteps + 1;
                    double tmp2 = tmp - input_means_tmp[j];
                    input_means_tmp[j]   += tmp2 / k;
                    input_sqmeans_tmp[j] += tmp2 * (tmp - input_means_tmp[j]); //(k - 1) / k * (tmp - input_means_tmp[j]);
                }
            }
            if (std_output) {
                float* outputs = readNcPatternArray(ncid, "targetPatterns", start, seqLength, output_size);
                //cout << "seq #" << i << " length = " << seqLength << endl;
                for (int t = 0; t < seqLength; ++t) {
                    for (int j = 0; j < output_size; ++j) {
                        float tmp = outputs[t * output_size + j];
                        //cout << "value = " << tmp << endl;
                        double k = t + total_timesteps + 1;
                        double tmp2 = tmp - output_means_tmp[j];
                        output_means_tmp[j]   += tmp2 / k;
                        output_sqmeans_tmp[j] += tmp2 * (tmp - output_means_tmp[j]);
                        //cout << "osqmeans[" << j << "] = " << output_sqmeans_tmp[j] << endl;
                    }
                }
                delete[] outputs;
            }
            delete[] inputs;
            total_timesteps += seqLength;
            start += seqLength;
        }
        float norm = 1.0 / float(total_timesteps);
        float norm2 = std::sqrt(float(total_timesteps) / float(total_timesteps - 1));
        for (int j = 0; j < input_size; ++j) {
            input_means[j] = float(input_means_tmp[j]);
            input_sds[j] = std::sqrt(input_sqmeans_tmp[j] / (total_timesteps - 1));
            cout << "input feature #" << j << ": mean = " << input_means[j] 
                 << " +/- " << input_sds[j] << endl;
        }
        if (std_output) {
            for (int j = 0; j < output_size; ++j) {
                output_means[j] = float(output_means_tmp[j]);
                output_sds[j] = std::sqrt(output_sqmeans_tmp[j] / (total_timesteps - 1));
                cout << "output feature #" << j << ": mean = " << output_means[j] 
                     << " +/- " << output_sds[j] << endl;
            }
        }

    }
    
    else {
        //string nc_file_norm = argv[2];
        int ncid_norm;
        if ((ret = nc_open(argv[2], NC_NOWRITE, &ncid_norm)) != NC_NOERR) {
            cerr << "Could not open '" << argv[2] << "': " << nc_strerror(ret) << endl;
            return 1;
        }
        cout << "Reading normdata from " << argv[2] << endl;
        try {
            readNcFloatArray(ncid_norm, "inputMeans", input_means, input_size);
            readNcFloatArray(ncid_norm, "inputStdevs", input_sds, input_size);
            for (int j = 0; j < input_size; ++j) {
                cout << "input feature #" << j << ": mean = " << input_means[j] 
                     << " +/- " << input_sds[j] << endl;
            }
            if (std_output) {
                readNcFloatArray(ncid_norm, "outputMeans", output_means, output_size);
                readNcFloatArray(ncid_norm, "outputStdevs", output_sds, output_size);
                for (int j = 0; j < output_size; ++j) {
                    cout << "output feature #" << j << ": mean = " << output_means[j] 
                         << " +/- " << output_sds[j] << endl;
                }
            }
        }
        catch (std::runtime_error err) {
            cerr << "Could not read normdata from " << argv[2] << ": " << err.what() << endl;
            return 1;
        }
    }
    
    // save normdata into nc file given by first argument
    cout << "save normdata" << endl;
    int input_means_varid;
    ret = createVarIfNotExists(ncid, "inputMeans", "inputPattSize", &input_means_varid, input_means);
    if (ret != NC_NOERR) {
       cerr << "ERROR saving inputMeans: " << nc_strerror(ret) << endl;
       return ret;
    }
    int input_stdevs_varid;
    ret = createVarIfNotExists(ncid, "inputStdevs", "inputPattSize", &input_stdevs_varid, input_sds);
    if (ret != NC_NOERR) {
       cerr << "ERROR saving inputStdevs: " << nc_strerror(ret) << endl;
       return ret;
    }
    if (std_output) {
        int output_means_varid;
        ret = createVarIfNotExists(ncid, "outputMeans", "targetPattSize", &output_means_varid, output_means);
        if (ret != NC_NOERR) {
           cerr << "ERROR saving inputMeans: " << nc_strerror(ret) << endl;
           return ret;
        }
        int output_stdevs_varid;
        ret = createVarIfNotExists(ncid, "outputStdevs", "targetPattSize", &output_stdevs_varid, output_sds);
        if (ret != NC_NOERR) {
           cerr << "ERROR saving inputMeans: " << nc_strerror(ret) << endl;
           return ret;
        }
    }

    // perform normalization on file pointed to by ncid
    int start = 0;
    for (int i = 0; i < total_sequences; ++i) {
        int seqLength = readNcIntArray(ncid, "seqLengths", i);
        int inputs_varid;
        float* inputs = readNcPatternArray(ncid, "inputs", start, seqLength, input_size, &inputs_varid);
        for (int t = 0; t < seqLength; ++t) {
            for (int j = 0; j < input_size; ++j) {
                int idx = t * input_size + j;
                inputs[idx] -= input_means[j];
                inputs[idx] /= input_sds[j];
            }
        }
        size_t starts[] = {start, 0};
        size_t counts[] = {seqLength, input_size};
        if ((ret = nc_put_vara_float(ncid, inputs_varid, starts, counts, inputs)) != NC_NOERR) {
            cerr << "Could not write standardized inputs: " << nc_strerror(ret) << endl;
            return ret;
        }
        if (std_output) {
            int outputs_varid;
            float* outputs = readNcPatternArray(ncid, "targetPatterns", start, seqLength, output_size, &outputs_varid);
            //cout << "seq #" << i << " length = " << seqLength << endl;
            for (int t = 0; t < seqLength; ++t) {
                for (int j = 0; j < output_size; ++j) {
                    int idx = t * output_size + j;
                    outputs[idx] -= output_means[j];
                    outputs[idx] /= output_sds[j];
                }
            }
            counts[1] = output_size;
            if ((ret = nc_put_vara_float(ncid, outputs_varid, starts, counts, outputs)) != NC_NOERR) {
                cerr << "Could not write standardized outputs: " << nc_strerror(ret) << endl;
                return ret;
            }
            delete[] outputs;
        }
        delete[] inputs;
        start += seqLength;
    }


    nc_close(ncid);
}
