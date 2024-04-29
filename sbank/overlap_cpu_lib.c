/*
 * Copyright (C) 2012  Nickolas Fotopoulos
 * Copyright (C) 2016  Stephen Privitera
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <sys/types.h>

/* ---------------- LAL STUFF NEEDED ----------------- */
int XLALPrintError(const char *fmt, ...);

typedef struct tagCOMPLEX8Vector {
     uint32_t length; /**< Number of elements in array. */
     float complex *data; /**< Pointer to the data array. */
} COMPLEX8Vector;

COMPLEX8Vector * XLALCreateCOMPLEX8Vector ( uint32_t length );
void XLALDestroyCOMPLEX8Vector ( COMPLEX8Vector * vector );

typedef struct tagCOMPLEX8FFTPlan COMPLEX8FFTPlan;

COMPLEX8FFTPlan * XLALCreateReverseCOMPLEX8FFTPlan( uint32_t size, int measurelvl );

int XLALCOMPLEX8VectorFFT (COMPLEX8Vector * restrict output, const COMPLEX8Vector * restrict input, const COMPLEX8FFTPlan *plan);

void XLALDestroyCOMPLEX8FFTPlan( COMPLEX8FFTPlan *plan );

/* ----------------------------------------------- */

typedef struct tagWS {
    size_t n;
    COMPLEX8FFTPlan *plan;
    COMPLEX8Vector *zf;
    COMPLEX8Vector *zt;
} WS;

WS *SBankCreateWorkspaceCache(void);

void SBankDestroyWorkspaceCache(WS *workspace_cache);

double _SBankComputeMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache);

double _SBankComputeRealMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache);

double _SBankComputeMatchMaxSkyLoc(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2);

double _SBankComputeMatchMaxSkyLocNoPhase(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2);

#define MAX_NUM_WS 32  /* maximum number of workspaces */
#define CHECK_OOM(ptr, msg) if (!(ptr)) { XLALPrintError((msg)); exit(-1); }

/*
 * set up workspaces
 */

WS *SBankCreateWorkspaceCache(void) {
    WS *workspace_cache = calloc(MAX_NUM_WS, sizeof(WS));
    CHECK_OOM(workspace_cache, "unable to allocate workspace\n");
    return workspace_cache;
}

void SBankDestroyWorkspaceCache(WS *workspace_cache) {
    size_t k = MAX_NUM_WS;
    for (;k--;) {
        if (workspace_cache[k].n) {
            XLALDestroyCOMPLEX8FFTPlan(workspace_cache[k].plan);
            XLALDestroyCOMPLEX8Vector(workspace_cache[k].zf);
            XLALDestroyCOMPLEX8Vector(workspace_cache[k].zt);
        }
    }
    free(workspace_cache);
}

static WS *get_workspace(WS *workspace_cache, const size_t n) {
    if (!n) {
        fprintf(stderr, "Zero size workspace requested\n");
        abort();
    }

    /* if n already in cache, return it */
    WS *ptr = workspace_cache;
    while (ptr->n) {
        if (ptr->n == n) return ptr;
        if (++ptr - workspace_cache > MAX_NUM_WS) return NULL;  /* out of space! */
    }

    /* if n not in cache, ptr now points at first blank entry */
    ptr->zf = XLALCreateCOMPLEX8Vector(n);
    CHECK_OOM(ptr->zf->data, "unable to allocate workspace array zf\n");
    memset(ptr->zf->data, 0, n * sizeof(float complex));

    ptr->zt = XLALCreateCOMPLEX8Vector(n);
    CHECK_OOM(ptr->zf->data, "unable to allocate workspace array zt\n");
    memset(ptr->zt->data, 0, n * sizeof(float complex));

    ptr->n = n;
    ptr->plan = XLALCreateReverseCOMPLEX8FFTPlan(n, 1);
    CHECK_OOM(ptr->plan, "unable to allocate plan");

    return ptr;
}

/* by default, complex arithmetic will call built-in function __muldc3, which does a lot of error checking for inf and nan; just do it manually */
static void multiply_conjugate(float complex * restrict out, float complex *a, float complex *b, const size_t size) {
    size_t k = 0;
    for (;k < size; ++k) {
        const float ar = crealf(a[k]);
        const float br = crealf(b[k]);
        const float ai = cimagf(a[k]);
        const float bi = cimagf(b[k]);
        __real__ out[k] = ar * br + ai * bi;
        __imag__ out[k] = ar * -bi + ai * br;
    }
}

static double abs_real(const float complex x) {
    const double re = crealf(x);
    return re;
}

static double abs2(const float complex x) {
    const double re = crealf(x);
    const double im = cimagf(x);
    return re * re + im * im;
}

/* interpolate the peak with a parabolic interpolation */
static double vector_peak_interp(const double ym1, const double y, const double yp1) {
    const double dy = 0.5 * (yp1 - ym1);
    const double d2y = 2. * y - ym1 - yp1;
    return y + 0.5 * dy * dy / d2y;
}

/*
 * Returns the match for two whitened, normalized, positive-frequency
 * COMPLEX8FrequencySeries inputs.
 */
double _SBankComputeMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws = get_workspace(workspace_cache, n);
    if (!ws) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }

    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws->zf->data, inj, tmplt, min_len);
    XLALCOMPLEX8VectorFFT(ws->zt, ws->zf, ws->plan); /* plan is reverse */

    /* maximize over |z(t)|^2 */
    float complex *zdata = ws->zt->data;
    size_t k = n;
    ssize_t argmax = -1;
    double max = 0.;
    for (;k--;) {
        double temp = abs2(zdata[k]);
        if (temp > max) {
            argmax = k;
            max = temp;
        }
    }
    if (max == 0.) return 0.;

    /* refine estimate of maximum */
    double result;
    if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        result = vector_peak_interp(abs2(zdata[argmax - 1]), abs2(zdata[argmax]), abs2(zdata[argmax + 1]));

    /* compute match */
    return 4. * delta_f * sqrt(result); 
}


/*
  Compute the overlap between a normalized template waveform h and a
  normalized signal proposal maximizing over the template h's overall
  amplitude. This is the most basic match function one can compute.
*/
double _SBankComputeRealMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws = get_workspace(workspace_cache, n);
    if (!ws) {
        XLALPrintError("out of space in the workspace_cache\n");
	exit(-1);
    }

    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws->zf->data, inj, tmplt, min_len);
    XLALCOMPLEX8VectorFFT(ws->zt, ws->zf, ws->plan); /* plan is reverse */

    /* maximize over |Re z(t)| */
    float complex *zdata = ws->zt->data;
    size_t k = n;
    double max = 0.;
    for (;k--;) {
	double temp = abs_real((zdata[k]));
	if (temp > max) {
	    max = temp;
	}
    }
    return 4. * delta_f * max;
}


/*
  Compute the overlap between a template waveform h and a signal
  proposal assuming only the (2,2) mode and maximizing over the
  template h's coalescence phase, overall amplitude and effective
  polarization / sky position. The function assumes that the plus and
  cross polarization hp and hc are both normalized to unity and that
  hphccorr is the correlation between these normalized components.
 */
double _SBankComputeMatchMaxSkyLoc(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws1 = get_workspace(workspace_cache1, n);
    if (!ws1) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws2 = get_workspace(workspace_cache2, n);
    if (!ws2) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }


    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws1->zf->data, hp, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws1->zt, ws1->zf, ws1->plan); /* plan is reverse */
    multiply_conjugate(ws2->zf->data, hc, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws2->zt, ws2->zf, ws2->plan);


    /* COMPUTE DETECTION STATISTIC */

    /* First start with constant values */
    double delta = 2 * hphccorr;
    double denom = 4 - delta * delta;
    if (denom < 0)
    {
        fprintf(stderr, "DANGER WILL ROBINSON: CODE IS BROKEN!!\n");
    }

    /* Now the tricksy bit as we loop over time*/
    float complex *hpdata = ws1->zt->data;
    float complex *hcdata = ws2->zt->data;
    size_t k = n;
    /* FIXME: This is needed if we turn back on peak refinement. */
    /*ssize_t argmax = -1;*/
    double max = 0.;
    for (;k--;) {
        double complex ratio = hcdata[k] / hpdata[k];
        double ratio_real = creal(ratio);
        double ratio_imag = cimag(ratio);
        double beta = 2 * ratio_real;
        double alpha = ratio_real * ratio_real + ratio_imag * ratio_imag;
        double sqroot = alpha*alpha + alpha * (delta*delta - 2) + 1;
        sqroot += beta * (beta - delta * (1 + alpha));
        sqroot = sqrt(sqroot);
        double brckt = 2*(alpha + 1) - beta*delta + 2*sqroot;
        brckt = brckt / denom;
        double det_stat_sq = abs2(hpdata[k]) * brckt;

        if (det_stat_sq > max) {
            /*argmax = k;*/
            max = det_stat_sq;
        }
    }
    if (max == 0.) return 0.;

    /* FIXME: For now do *not* refine estimate of peak. */
    /* double result;
    if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        result = vector_peak_interp(abs2(zdata[argmax - 1]), abs2(zdata[argmax]), abs2(zdata[argmax + 1])); */

    /* Return match */
    return 4. * delta_f * sqrt(max);
}

double _SBankComputeMatchMaxSkyLocNoPhase(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws1 = get_workspace(workspace_cache1, n);
    if (!ws1) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws2 = get_workspace(workspace_cache2, n);
    if (!ws2) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }


    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws1->zf->data, hp, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws1->zt, ws1->zf, ws1->plan); /* plan is reverse */
    multiply_conjugate(ws2->zf->data, hc, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws2->zt, ws2->zf, ws2->plan);


    /* COMPUTE DETECTION STATISTIC */

    /* First start with constant values */
    double denom = 1. - (hphccorr*hphccorr);
    if (denom < 0)
    {
        fprintf(stderr, "DANGER WILL ROBINSON: CODE IS BROKEN!!\n");
    }

    /* Now the tricksy bit as we loop over time*/
    float complex *hpdata = ws1->zt->data;
    float complex *hcdata = ws2->zt->data;
    size_t k = n;
    /* FIXME: This is needed if we turn back on peak refinement. */
    /*ssize_t argmax = -1;*/
    double max = 0.;
    double det_stat_sq;

    for (;k--;) {
        det_stat_sq = creal(hpdata[k])*creal(hpdata[k]);
        det_stat_sq += creal(hcdata[k])*creal(hcdata[k]);
        det_stat_sq -= 2*creal(hpdata[k])*creal(hcdata[k])*hphccorr;

        det_stat_sq = det_stat_sq / denom;

        if (det_stat_sq > max) {
            /*argmax = k;*/
            max = det_stat_sq;
        }
    }
    if (max == 0.) return 0.;

    /* FIXME: For now do *not* refine estimate of peak. */
    /* double result;
    if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        result = vector_peak_interp(abs2(zdata[argmax - 1]), abs2(zdata[argmax])
, abs2(zdata[argmax + 1])); */

    /* Return match */
    return 4. * delta_f * sqrt(max);
}
