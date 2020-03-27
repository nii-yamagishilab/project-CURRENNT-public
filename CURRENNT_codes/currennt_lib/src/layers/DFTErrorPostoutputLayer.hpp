/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2018 - 2019
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef LAYERS_DFTERRORPOSTOUTPUT_HPP
#define LAYERS_DFTERRORPOSTOUTPUT_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    template <typename TDevice>
    class DFTPostoutputLayer : public PostOutputLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename TDevice::complex_vector  fft_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
	
    private:

	struct struct_DFTData{
	    
	    // whether this DFT buffer will be used?
	    bool               m_valid_flag;

	    // spectra amplitude Error
	    real_t             m_specError;
	    
	    // phase error
	    real_t             m_phaseError;
	    
	    // complex-valued spectra error
	    real_t             m_resError;
	    
	    // real-valued spectra error
	    real_t             m_realSpecError;

	    // lpc error
	    real_t             m_lpcError;

	    // spec error for hidden layers
	    real_t             m_specErrorOthers;
	    
	    // configuration for short-time analysis
	    int                m_frameLength;
	    int                m_frameShift;
	    int                m_windowType;
	    int                m_windowTypePhase;
	    int                m_fftLength;
	    
	    int                m_frameNum;	    
	    int                m_fftBinsNum;

	    int                m_lpcOrder;
	    
	    // dft buffers
	    real_vector        m_fftSourceFramed;    // source signal being framed
	    fft_vector         m_fftSourceSigFFT;    // FFT of framed source signal (STFT)
	    
	    real_vector        m_fftTargetFramed;    // target signal begin framed
	    fft_vector         m_fftTargetSigFFT;    // FFT of framed target signal (STFT)
	    
	    real_vector        m_fftDiffData;        // waveform grad w.r.t spec-amp-distance
	    real_vector        m_fftDiffFramed;      // framed waveform grad w.r.t spec-amp-distance
	    fft_vector         m_fftDiffSigFFT;      // to store the spec-amp-distance

	    // gradietns buffer for phase and complex-valued spectral distances
	    real_vector        m_fftDiffDataPhase;
	    real_vector        m_fftResData;

	    // for spectral mask
	    real_vector        m_fftMaskSignalFramed;
	    fft_vector         m_fftMaskFFT;
	    
	    // for real-valued spectral 
	    int                m_fftLengthRealSpec;
	    int                m_fftBinsNumRealSpec;
	    
	    real_vector        m_fftSourceFramedRealSpec;
	    fft_vector         m_fftSourceSigFFTRealSpec;
	    
	    real_vector        m_fftTargetFramedRealSpec;
	    fft_vector         m_fftTargetSigFFTRealSpec;
	    
	    real_vector        m_fftDiffDataRealSpec;
	    real_vector        m_fftDiffFramedRealSpec;
	    fft_vector         m_fftDiffSigFFTRealSpec;	


	    // for LPC analysis
	    real_vector        m_autoCorrSrc; // buffer for auto-correlation 
	    real_vector        m_lpcCoefSrc;  // buffer for LPC coefficients
	    real_vector        m_lpcErrSrc;   // buffer for LPC error
	    real_vector        m_refCoefSrc;  // buffer for reflect coefficients
	    real_vector        m_lpcResSrc;   // buffer for LPC residual
	    
	    real_vector        m_autoCorrTar; // buffer for auto-correlation 
	    real_vector        m_lpcCoefTar;  // buffer for LPC coefficients
	    real_vector        m_lpcErrTar;   // buffer for LPC error
	    real_vector        m_refCoefTar;  // buffer for reflect coefficients
	    real_vector        m_lpcResTar;   // buffer for LPC residual
	    
	    real_vector        m_lpcGrad;  // buffer to store the gradients
	    
	    // for additional spec-amp-distances on hidden layers
	    real_vector        m_specGrad_others;
	    real_vector        m_specGrad_tmpBuf;
	};

	/*
	  Error =  m_beta * waveform_MSE + m_gamma * spectral_amplitude_MSE + m_zeta * phase_MSE
	  + m_eta * residual_signal_spectral_amplitude + m_kappa * real_spectrum_amp 
	  + m_tau * lpc_error + m_iota * spectral_amplitude_for_hidden_layers
	 */

	real_t             m_beta;              // Weight for waveform MSE
	real_t             m_gamma;             // Weight for DFT amplitude 
	real_t             m_zeta;              // Weight for DFT phase
	real_t             m_eta;               // Weight for residual spectrum amplitude
	real_t             m_kappa;             // Weight for realvalued-spectrum ampltiude
	real_t             m_tau;               // Weight for error in LPC-domain
	real_t             m_iota;              // weight for DFT amplitude distances for hidden layers
	
	real_t             m_mseError;          //

	
	int                m_preEmphasis;       // Whether preEmphasis the natural speech?
	int                m_specDisType;       // Type of spectral amplitude distance
	int                m_phaseDisType;      // Type of phase distance
	
	int                m_realSpecType;      // Type of real-valued spectrum
	int                m_realSpecDisType;   // Type pf real-valued spectrum distance	
	int                m_lpcErrorType;      // Type of LPC error
	
	int                m_lpcGain;           // whether LPC Gain should be used

	real_t             m_floor_log_spec_amp; // floor for spec_amp

	
	// support for evaluation on signals from multiple layers
	int                m_separate_excitation_loss;
	int                m_otherSignalInputLayer_num;
	std::string        m_otherSignalInputLayers_str;        // string of additional input layers
	std::vector<std::string> m_otherSignalInputLayers_names; // string buffer of additional input layers
	std::vector<Layer<TDevice>*> m_otherSignalInputLayers_ptr; // pointer to previous layers

	// support to receive input from sine
	std::string        m_sineInputLayer_str;
	Layer<TDevice>*    m_sineInputLayer_ptr;

      	// support to receive input from sine
	std::string        m_f0InputLayer_str;
	Layer<TDevice>*    m_f0InputLayer_ptr;

	// data structure for DFT analysis
	std::vector<struct_DFTData> m_DFTDataBuf;
	
	
	// support for harmonic + noise model (not used anymore)
	int                m_hnm_flag;
	int                m_noiseTrain_epoch;
	Layer<TDevice>*    m_noiseOutputLayer;
	Layer<TDevice>*    m_f0InputLayer;
	std::string        m_noiseOutputLayerName;
	std::string        m_f0InputLayerName;
	real_t             m_f0DataM;
	real_t             m_f0DataS;

	int                m_modeMultiDimSignal;
	real_vector        m_modeChangeDataBuf;	
	
	// methods for initialization
	void __loadOpts(const helpers::JsonValue &layerChild);
	void __cleanDFTError(struct_DFTData &dftBuf);
	void __initDFTBuffer(struct_DFTData &dftBuf);	
	void __configDFTBuffer(struct_DFTData &dftBuf,
			       const int fftLength, const int frameLength,
			       const int frameShift, const int windowType,
			       const int windowTypePhase, const int lpcOrder);
	
	// methods for utilities
	int  __vSize();
	int  __vMaxSeqLength();
	int  __vCurMaxSeqLength();

	// methods for waveform pre-emphasis
	void __preEmphasis(const int timeLength);
	void __deEmphasis(const int timeLength);

	// methods for flattenning multi-dimensional signal
	void __flattenMultiDimSignalForward(const int timeLength);
	void __flattenMultiDimSignalBackward(const int timeLength);

	// obsolete methods for HNM training
	void __hnmSpecialForward(const int timeLength, const int nnState);
	void __hnmSpecialBackward(const int timeLength, const int nnState);

	// methods for waveform MSE
	real_t __waveformMseForward(const int timeLength);
	void __waveformMseBackward(const int timeLength);

	
	// spectral amplitude distance
        real_t __specAmpDistance(struct_DFTData &dftBuf, const int timeLength);
	
	// phase distance
	real_t __specPhaDistance(struct_DFTData &dftBuf, const int timeLength);
	
	// complex-valued spectral distance
	real_t __specResAmpDistance(struct_DFTData &dftBuf, const int timeLength);
	
	// real-valued spectral distance
	real_t __specRealAmpDistance(struct_DFTData &dftBuf, const int timeLength);

	// LPC error
	real_t __lpcError(struct_DFTData &dftBuf, const int timeLength);

	// spec errors for other hidden layers
	real_t __specAmpDistanceOthers(struct_DFTData &dftBuf, const int timeLength);
	
	// a wrapper to wrap all the distances
	real_t __specDistance_wrapper(struct_DFTData &dftBuf, const int timeLength);

	// a wrapper to accumulate gradients
	void __specAccumulateGrad(struct_DFTData &dftBuf, const int timeLength);

	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         */
        DFTPostoutputLayer(
            const helpers::JsonValue &layerChild, 
            Layer<TDevice> &precedingLayer,
	    int             maxSeqLength,
	    int             layerID);

        /**
         * Destructs the Layer
         */
        virtual ~DFTPostoutputLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see PostOutputLayer::calculateError()
         */
        virtual real_t calculateError();

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int nnState);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int timeStep, const int nnState);

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int nnState);

	virtual void computeBackwardPass(const int timeStep, const int nnState);
	
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;


	// load the target data from the target layer
	virtual void linkTargetLayer(Layer<TDevice> &targetLayer);

	// 
	std::vector<int> dependLayerIDs();
    };

} // namespace layers

#endif
