import os
import sys
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.modeling import models, fitting
from PyAstronomy import pyasl

import scipy as sp
from astropy.stats import sigma_clip

from astropy import units as u
#import ispec
#import example as ex

from io import BytesIO                              #### added to save as pdf
# from reportlab.pdfgen import canvas                 #### added to save as pdf
# from reportlab.graphics import renderPDF            #### added to save as pdf
# from reportlab.lib.utils import ImageReader         #### added to save as pdf
#from reportlab.lib.units import inch, cm            #### added to save as pdf
from svglib.svglib import svg2rlg                   #### added to save as pdf
# from reportlab.platypus.tables import Table, SimpleDocTemplate, TableStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

#
# plt.rcParams['figure.figsize'] = 3.5, 3.

print('test')

def main():
    script = sys.argv[0]
    filename = sys.argv[1]
    model_filename = sys.argv[2]
    action = sys.argv[3]
    assert action in ['ccf', 'none']

    try:
        process(filename, model_filename, action)
    except Exception as e:
        print (e)


def process(f, t, action):
    data = pf.getdata(f) # Get the data
    hdr = pf.getheader(f)
    print(hdr['BVC'])
    if action == 'ccf':
        print('Normalizing spectra')

        plt.figure()
        plt.plot(data[4,30,:], data[0,30,:])

        for i in range(len(data[0,:,0])):
            blaze_corr = np.divide(data[0,i,:],data[2,i,:], out=np.zeros_like(data[0,i,:]), where=data[2,i,:]!=0) # blaze-correction
            print(np.median(blaze_corr))
            median_nor = blaze_corr/np.median(blaze_corr) ### rough normalization by divding with median    running mean
            fit = np.polyfit(data[4,i,:], median_nor, 1)  # fitting linear function
            normalized = np.poly1d(fit) #creating linear fit to spectra
            tmp = median_nor/normalized(data[4,i,:])                            # normalizing
            "binning the data, and running a cubic spline over each bin"
            statistic, bin_edges, binnumber = sp.stats.binned_statistic(data[4,i,:], tmp,statistic='median',bins=10)
            bin_edges = bin_edges+0.5*(bin_edges[1]-bin_edges[0])
            y_CubicSpline_fit = sp.interpolate.CubicSpline(bin_edges[0:10],statistic)

            data[0,i,:] = tmp/y_CubicSpline_fit(data[4,i,:])

        print('preparing model')

#        template_data = pf.getdata(t) # Get the data
#        template_hdr = pf.getheader(t)
        temp_wave, temp_flux = pyasl.read1dFitsSpec(t)


#        temp_wave, temp_flux = ex.synthesize_spectrum(code="spectrum",teff = 6399., logg = 4.26, MH = 0.26, vsini = 14.27, limb_darkening_coeff = 0.6, resolution = 100000, wave_step = 0.01, regions = None, wave_base = data[4,0,0]/10-50, wave_top = data[4,-1,-1]/10+50)

        print(type(temp_wave))
        print(type(temp_flux))

        print(temp_wave[0], temp_wave[-1])
        print(data[4,0,0]*0.1, data[4,-1,-1]*0.1)

        print('running ccf')
        rv = np.empty([51, 200])
        cc = np.empty([51, 200])
        g = []
        for i in range(len(data[0,:,0])):
            rv[i,:], cc[i,:] = pyasl.crosscorrRV(data[4,i,:]*0.1, 1-data[0,i,:], temp_wave, 1-temp_flux, -100., 100., 1, skipedge=10)
            #rv[i,:], cc[i,:] = pyasl.crosscorrRV(temp_wave, 1-temp_flux, data[4,i,:]*0.1, 1-data[0,i,:], -100., 100., 1, skipedge=100)
            rv[i,:] = rv[i,:]+hdr['BVC']
            g_init = models.Gaussian1D(amplitude=1., mean=12, stddev=1.)
            fit_g = fitting.LevMarLSQFitter()
            x = fit_g(g_init, rv[i, :], cc[i, :])
            print(x.mean)
            print(x.mean[0])
            g.append(x.mean[0])
        print(np.median(g))



        # plt.figure(2)
        # for i in range(10):
        #     plt.plot(rv[i].T,10*(i+1)+cc[i].T)
        # plt.figure(3)
        # for i in range(10):
        #     plt.plot(rv[i+10].T,(i+1)+cc[i+10].T)
        # plt.figure(4)
        # for i in range(10):
        #     plt.plot(rv[i+20].T,(i+1)+cc[i+20].T)
        # plt.figure(5)
        # for i in range(10):
        #     plt.plot(rv[i+30].T,(i+1)+cc[i+30].T)
        # plt.figure(6)
        # for i in range(11):
        #     plt.plot(rv[i+40].T,(i+1)+cc[i+40].T)
        #
        #
        # plt.figure(7)
        # for i in range(10):
        #     plt.plot((i+1)+data[0,i,:].T)
        # plt.figure(8)
        # for i in range(10):
        #     plt.plot((i+1)+data[0,i+10,:].T)
        # plt.figure(9)
        # for i in range(10):
        #     plt.plot((i+1)+data[0,i+20,:].T)
        # plt.figure(10)
        # for i in range(10):
        #     plt.plot((i+1)+data[0,i+30,:].T)
        # plt.figure(11)
        # for i in range(11):
        #     plt.plot((i+1)+data[0,i+40,:].T)

        fig1 = plt.figure(11)
        for i in range(len(g)):
            plt.style.use('ggplot')
            plt.plot(rv[i].T,cc[i].T, 'b', alpha=0.1)
            plt.axvline(x = np.median(g), color='r', alpha=0.1, linewidth=1.)

        fig2 = plt.figure(12)
        plt.style.use('ggplot')
        plt.plot(rv[0].T,np.mean(cc, axis=0), 'b', alpha=0.1)
        plt.axvline(x = np.mean(g[:-1]), color='r', alpha=0.1, linewidth=1.)

        bv_corr = hdr['BVC']/300000*np.mean(data[4,35,:])
        print(bv_corr)

        plt.figure()
        plt.plot(range(50), g[0:-1])

        fig3 = plt.figure()
        ax1 = fig3.add_subplot(4,1,1)
        ax1.plot(data[4,35,:]+bv_corr, data[0,35,:])
        ax1.plot(temp_wave*10,temp_flux)
        ax2 = fig3.add_subplot(4,1,2)
        ax2.plot(data[4,46,:]+bv_corr, data[0,46,:])
        ax2.plot(temp_wave*10,temp_flux)
        ax3 = fig3.add_subplot(4,1,3)
        ax3.plot(data[4,21,:]+bv_corr, data[0,21,:])
        ax3.plot(temp_wave*10,temp_flux)
        ax4 = fig3.add_subplot(4,1,4)
        ax4.plot(data[4,50,:]+bv_corr, data[0,50,:])
        ax4.plot(temp_wave*10,temp_flux)


        "for PDF creation"
        print(hdr)
        imgdata1 = BytesIO()                         #### added to save as pdf
        fig1.savefig(imgdata1, format='svg')          #### added to save as pdf
        imgdata1.seek(0)  # rewind the data          #### added to save as pdf
        Image1 = svg2rlg(imgdata1)                #### added to save as pdf



        sample_style_sheet = getSampleStyleSheet()
        my_doc = SimpleDocTemplate('test2.pdf')
        flowables = []
        title = "Observations"
        paragraph_1 = Paragraph(title, sample_style_sheet['Heading4'])
        data = [['RA:', hdr['OBJ-RA'], 'Target', hdr['OBJ-NAME']],
        ['DEC:', hdr['OBJ-DEC'],'Observatory', hdr['OBSERVAT']],
        ['V-Mag',hdr['OBJ-MAG'],'EXPTIME [s]',hdr['EXPTIME']],
        ['BVC [km/s]',hdr['BVC'], 'OBS-MODE', hdr['OBS-MODE']]]
        t = Table(data)
        paragraph_2 = Paragraph("Template-spectrum", sample_style_sheet['Heading4'])


        t.setStyle(TableStyle([('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        #('TEXTCOLOR',(1,0),(1,-1), colors.green),
        ('LINEBEFORE',(2,0),(2,-1),1,colors.black),
        ('ALIGN',(0,0),(0,-1),'LEFT'),
        ('ALIGN',(1,0),(1,-1),'RIGHT'),
        ('ALIGN',(2,0),(2,-1),'LEFT'),
        ('ALIGN',(3,0),(3,-1),'RIGHT')
        ]))
#
        flowables.append(paragraph_1)
        flowables.append(t)
        flowables.append(paragraph_2)
        flowables.append(t)
        flowables.append(Image1)
        flowables.append(t)

        my_doc.build(flowables)


#
#         imgdata2 = BytesIO()                         #### added to save as pdf
#         fig2.savefig(imgdata2, format='svg')          #### added to save as pdf
#         imgdata2.seek(0)  # rewind the data          #### added to save as pdf
#
#         Image1 = svg2rlg(imgdata1)                #### added to save as pdf
#         Image2 = svg2rlg(imgdata2)                #### added to save as pdf
#         c = canvas.Canvas('test.pdf')
#         c.drawCentredString(2*inch, 2*inch,"text")
# #        c.addOutlineEntry(title, key, level=0, closed=None)
#         renderPDF.draw(Image1,c, 0*cm, 5*inch)
#         renderPDF.draw(Image2,c, 4*inch, 5*inch)
#         c.drawString(100, 750, title)
#         renderPDF.draw(table,c, 0*cm, 0*cm)
#         c.save()                                    #### added to save as pdf

#        plt.show()

    elif action == 'none':
        print(np.shape(data))
        print(hdr)
        plt.plot( data[4,29,:], data[0,29,:] )
    plt.show()

if __name__ == '__main__':
    main()
