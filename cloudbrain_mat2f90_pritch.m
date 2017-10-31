clear; close all;
target_names   = {'SPDT','SPDQ'}; 
resultsdir = './results';
unix (sprintf ('mkdir -p %s',resultsdir)); 

for ktarget = 1:4
    target_name = char(target_names(ktarget)); 

    % These numbers distinguish which neural net we are talking about, the
    % relevant dimensions of # neurons and # hidden layers.
    NbNeurons       = 10;
    NbHiddenLayers  = 1;
    seednumber      = 80;

    checkpointdir= '/Users/Pritchard/Dropbox/CHECKPOINTS';
    FILE_PREFIX = 'ANN_IN__PS_QAP_TAP_OMEGA_SHFLX_LHFLX_OUT__';
    fields = {'PS','QAP','TAP','OMEGA','SHFLX','LHFLX'};
    input_names    = {'PS','QAP','TAP','OMEGA','SHFLX','LHFLX'};
    output_fct     = 'purelin';

    filename = [checkpointdir '/' FILE_PREFIX target_name '_NbNeurons=' num2str(NbNeurons) '_NbHiddenLayers=' num2str(NbHiddenLayers) '_seed=' num2str(seednumber) '_' output_fct '.mat'];

    if ~exist(filename)
        disp (filename); 
        error ('Cannot load file'); 
    end
    
    load(filename,'net');

    iw = net.IW;  % containing input layers weight matrix
    lw = net.LW;  % containing hidden layers weight matrices
    b  = net.b;   % containing bias vectors

    % Figure out the f90 dimension sizes for the net:

    hiddenlayerSize = size(b{1,1}); 
    if (hiddenlayerSize ~= NbNeurons)
        error ('unexpected'); 
    end
    outputlayerSize = size (b{2,1}); 
    if (outputlayerSize ~= 28)
        error ('unexpected'); 
    end
    inputlayerSize = size (iw{1,1},2);
    if (inputlayerSize ~= 93)
        error ('unexpected'); 
    end


    % Create the f90 hardwired neural net source code file
    fid = fopen (sprintf ('%s/cloudbrain_module.F90',resultsdir),'w');
    if (ktarget == 1)
        % (just the first time) write source for module definition, 
        % its needed links to CAM data structures, and the internal 
        % neural net dim size parameter definitions.
        
        fprintf (fid,'#include <misc.h>\n'); 
        fprintf (fid,'#include <params.h>\n\n'); 
        fprintf (fid,'module cloudbrain_module\n'); 

        fprintf (fid,'use shr_kind_mod,    only: r8 => shr_kind_r8\n'); 
        fprintf (fid,'use ppgrid,          only: pcols, pver, pverp\n'); 
        fprintf (fid,'use history,         only: outfld, addfld, add_default, phys_decomp\n'); 
        fprintf (fid,'use nt_FunctionsModule, only: nt_tansig\n'); 
        fprintf (fid,'use physconst,       only: gravit,cpair\n'); 
        fprintf (fid,'implicit none\n\n'); 
        fprintf (fid,'save\n\n'); 
        fprintf (fid,'private                         ! Make default type private to the module\n'); 

        fprintf(fid,'integer,parameter :: hiddenlayerSize = %d\n',inputlayerSize); 
        fprintf(fid,'integer,parameter :: outputlayerSize = %d\n',outputlayerSize); 
        fprintf(fid,'integer, parameter :: nbhiddenlayers = %d\n',NbHiddenLayers); 
        fprintf(fid,'integer,parameter :: inputlayerSize = %d\n\n',inputlayerSize); 

        fprintf(fid,'public cloudbrain\n\n'); 
        fprintf(fid,'contains: \n\n'); 
    end
    
    % For each target, write the subroutine defining its neural net as read
    % from the matlab data file in memory.
    
    fprintf (fid,'subroutine define_neuralnet_%s (mu_in,std_in,weights_input,bias_input,bias_output,weights_output)\n',target_name); 
    fprintf (fid,'real(r8), intent(out) :: mu_in(inputlayerSize)\n'); 
    fprintf (fid,'real(r8), intent(out) :: std_in(inputlayerSize)\n');
    fprintf (fid,'real(r8), intent(out) :: bias_input(hiddenlayerSize)\n'); 
    fprintf (fid,'real(r8), intent(out) :: bias_output(outputlayerSize)\n'); 
    fprintf (fid,'real(r8), intent(out) :: weights_input(hiddenlayerSize,inputlayerSize)\n'); 
    fprintf (fid,'real(r8), intent(out) :: weights_output(outputlayerSize,hiddenlayerSize)\n');
    
    bias_input = b{1,1}; 
    str = 'bias_input(:) = (/';
    for k=1:hiddenlayerSize
        str = [str sprintf('%10e, ',bias_input(k))]; 
    end
    str = [str(1:end-2) ' /)\n'];
    fprintf (fid,str);     
    
    bias_output =  b{2,1};
    str = 'bias_output(:) = (/';
    for k=1:outputlayerSize        
        str = [str sprintf('%10e, ',bias_output(k))]; 
    end
    str = [str(1:end-2) ' /)\n'];
    fprintf (fid,str); 
   
    weights_input = iw{1,1}; 
    for kneuron=1:hiddenlayerSize
        str = sprintf ('weights_input(%d,:) = (/',kneuron); 
        for k=1:inputlayerSize
            str = [str sprintf('%10e, ',weights_input(kneuron,k))];
        end
        str = [str(1:end-2) ' /)\n'];
        fprintf (fid,str); 
    end
    
    weights_output = lw{2,1}; 
    for kout=1:outputlayerSize
        str = sprintf ('weights_output(%d,:) = (/',kout); 
        for k=1:hiddenlayerSize
            str = [str sprintf('%10e, ',weights_output(kout,k))];
        end
        str = [str(1:end-2) ' /)\n'];
        fprintf (fid,str); 
    end

    fprintf (fid,'\n\n');
    fprintf (fid,'end subroutine define_neuralnet_%s \n\n\n',target_name); 

end

fprintf (fid,'end module cloudbrain\n'); 
fclose (fid);
