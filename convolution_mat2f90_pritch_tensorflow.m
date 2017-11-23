% Produces CAM3-compatible f90 implementation of the convo neural network that does
% joint prediction of SPDT,SPDQ.
% By Mike Pritchard, UCI. 
clear; close all;

% The high level algorithm here is:
%   i) train the neural net using tensorflow, make tf logs & checkpoints.
%   ii) use python script extract_weights_txt.py to mine checkpoint to produce matrices
%   defining the convo net, in .mat format.
%   iii) run this script to make a fortran subroutine that can convert
%   input to outputs using the convo net design.

% Note this script depends on .mat file outputs from extract_weights_txt.py
% assumed to exist in the current directory.

% Also depends on normalization data consistent with the training, stored here: 
meanfile = './normalizations/SPCAM_mean.nc';
stdfile = './normalizations/SPCAM_std.nc';

% parameters, need to set consistent with the net ---
% (warning convo filter length 3 is hardwired)
output_varnames   = {'SPDT','SPDQ'}; 
output_units = {'W/kg','W/kg'}; 
output_varnames_is3d = [1 1];
input_varnames = {'TAP','QAP','OMEGA','SHFLX','LHFLX','dTdt_adiabatic','dQdt_adiabatic'};
input_varnames_is3d = [1 1 1 0 0 1 1];
nbneurons       = 32;
nbhiddenlayers  = 6;
nlev = 21; % Only the bottom this-many levels are used in the neural net.
% end parameters ---

nchannel_in = length(input_varnames); % Inputs to the N-N.
nchannel_out = length(output_varnames); 
resultsdir = './results';
unix (sprintf ('mkdir -p %s',resultsdir)); 

% Create the f90 hardwired neural net source code file
% (just the first time) write source for module definition, 
% its needed links to CAM data structures, and the internal 
% neural net dim size parameter definitions.
fid = fopen (sprintf ('%s/cloudbrain_convo_module.F90',resultsdir),'w');

fprintf (fid,'#include <misc.h>\n'); 
fprintf (fid,'#include <params.h>\n\n'); 
fprintf (fid,'module cloudbrain_convo_module\n'); 

fprintf (fid,'use shr_kind_mod,    only: r8 => shr_kind_r8\n'); 
fprintf (fid,'use ppgrid,          only: pcols, pver, pverp\n'); 
fprintf (fid,'use history,         only: outfld, addfld, add_default, phys_decomp\n'); 
fprintf (fid,'use physconst,       only: gravit,cpair\n'); 
fprintf (fid,'implicit none\n\n'); 
fprintf (fid,'save\n\n'); 
fprintf (fid,'private                         ! Make default type private to the module\n'); 

fprintf(fid,'integer,parameter :: nbneurons = %d\n',nbneurons); 
fprintf(fid,'integer, parameter ::nbhiddenlayers = %d\n',nbhiddenlayers); 
fprintf(fid,'integer,parameter :: nlev = %d ! Only the bottom this-many levels are used in the neural net.\n',nlev); % We used 21 of the 30 layers for the net. 
fprintf(fid,'integer,parameter :: nchannel_in = %d ! Input variables \n\n',nchannel_in); 
fprintf(fid,'integer,parameter :: nchannel_out = %d ! Output variables \n\n',nchannel_out); 

fprintf(fid,'public cloudbrain\n\n'); 
fprintf(fid,'contains \n\n'); 
        
% The master subroutine:
str = 'subroutine cloudbrain_convo(';
% Declare the input variables:
for k=1:length(input_varnames)
    str = [str char(input_varnames(k)) ', '];
end
for k=1:length(output_varnames)
   str = [str char(output_varnames(k)) ', '];  
end
str = [str(1:end-2) ')\n\n'];
fprintf (fid,str);
for k=1:length(input_varnames)
    if (input_varnames_is3d(k))
        fprintf (fid,'real(r8), intent(in) :: %s(pver)\n',char(input_varnames(k))); 
    else
        fprintf (fid,'real(r8), intent(in) :: %s\n',char(input_varnames(k))); 
    end
end
for k=1:length(output_varnames)
    if (output_varnames_is3d(k))
        fprintf (fid,'real(r8), intent(out) :: %s(pver) ! %s\n',char(output_varnames(k)),char(output_units(k))); 
    else
        fprintf (fid,'real(r8), intent(out) :: %s ! %s \n',char(output_varnames(k)),char(output_units(k))); 
    end
end
fprintf (fid,'real(r8) :: input(nlev,nchannel_in)\n'); 
fprintf (fid,'real(r8) :: input_padded(nlev+2,nchannel_in)\n'); 
fprintf (fid,'real(r8) :: interior_input_padded(nlev+2,nbneurons)\n'); 
fprintf (fid,'real(r8) :: new_state (nlev,nbneurons)\n');

fprintf (fid,'real(r8) :: conv2d_input_kernel(3,nchannel_in,nlev)\n'); 
fprintf (fid,'real(r8) :: conv2d_input_bias(nbneurons)\n'); 
fprintf (fid,'real(r8) :: conv2d_current_interior_kernel(3,nbneurons,nlev)\n'); 
fprintf (fid,'real(r8) :: conv2d_current_interior_bias(nbneurons)\n'); 
fprintf (fid,'real(r8) :: conv2d_outer_kernel (nbneurons,nchannel_out)\n'); 
fprintf (fid,'real(r8) :: conv2d_outer_bias (nchannel_out)\n'); 
fprintf (fid,'real(r8) :: out_state (nlev,nchannel_out)\n'); 

fprintf (fid,'real(r8) :: meanvect(nlev),stdvect(nlev) ! tmp buffers for var-specific normalization vectors. \n'); 
fprintf (fid,'integer :: ilayer,k,kchan,jnewchan,kfilt \n\n'); 

fprintf (fid,'! ------- neural net input matrix & normalization ----\n'); 
% Assemble the 21 x nchannel_in input matrix needed for the convo network.
for kchannelin=1:nchannel_in
    % normalization data for the current variable i.e. input channel.
    auxmean = ncread(meanfile,char(input_varnames(kchannelin)));    
    auxstd = ncread(stdfile,char(input_varnames(kchannelin)));        
    if ~(input_varnames_is3d(kchannelin))
        fprintf (fid,'input(:,kchannelin) = (%s-%f)/%f\n',char(input_varnames(k)),auxmean,auxstd); % link to subroutine input var data
    else
        mean_str = sprintf ('%f',auxmean(1)); 
        std_str = sprintf ('%f',auxstd(1)); 
        for kk=2:length(auxmean)
            mean_str = sprintf ('%s,%f',mean_str,auxmean(kk)); 
            std_str = sprintf ('%s,%f',mean_str,auxstd(kk)); 
        end           
        fprintf (fid,'meanvect(:)= (/ %s /) \n',mean_str); 
        fprintf (fid,'stdvect(:)= (/ %s /) \n',std_str); 
        fprintf (fid,'input(:,%d)= (%s((pver-nlev+1):pver) - meanvect)/stdvect \n',kchannelin,char(input_varnames(kchannelin)));
    end
end
% 0) First layer... 21 x 7 input --> 23 x 7 (padded input) --> 21 x 32 (after
% filter application) --> activation.

fprintf (fid,'! ------- input layer, define underlying matrices: ----\n'); 
fprintf (fid,'call get_inputlayer_data (conv2d_input_bias,conv2d_input_kernel)\n'); 

fprintf (fid,'\n! ------- input layer, pad and do the magic:\n'); 
fprintf ('input_padded(2:(nlev+1),:) = input(:,:)\n')
fprintf (fid,'input_padded(1,:) = input(1,:)\n');
fprintf (fid,'input_padded((nlev+2),:) = input(nlev,:)\n');
fprintf (fid,'! ---- Go from 23x7 padded input to 21x32 for first dense layer\n'); 
fprintf (fid,'new_state(:,:) = 0.\n');
fprintf (fid,'do jnewchan = 1,nbneurons then\n'); % New channels
fprintf (fid,'  do kchan = 1,nchannel_in then\n'); 
fprintf (fid,'    do k = 2,nlev+1 then\n')
fprintf (fid,'      do kfilt = 1,3 then\n')
fprintf (fid,'        new_state(k-1,jnewchan) = new_state(k-1,jnewchan) + input_padded(k+kfilt-2,kchan)*conv2d_input_kernel(kfilt,kchan,jnewchan)\n'); 
fprintf (fid,'      end do\n');
fprintf (fid,'    end do\n');
fprintf (fid,'  end do\n');
fprintf (fid,'! --- Apply bias and then leaky relu activation function:\n'); 
fprintf (fid,'  do k=1,nlev then\n'); 
fprintf (fid,'    new_state(k,jnewchan) = new_state(k,jnewchan) + conv2d_input_bias(jnewchan)\n'); 
fprintf (fid,'    if (new_state(k,jnewchan) .lt. 0.) then\n'); 
fprintf (fid,'      new_state(k,jnewchan) = new_state(k,jnewchan)*0.3\n'); 
fprintf (fid,'    end if\n'); 
fprintf (fid,'  end do\n');
fprintf (fid,'end do\n');

fprintf (fid,'! ---- NOW do the same as above but for an outer loop over all the hidden layers:\n'); 
fprintf (fid,'! ---- in each step, we go from a 21x32 --> 23x32 padded --> (net matrices) --> the next 21x32 --> its bias, activation :\n'); 

fprintf (fid,'do ilayer = 1,nbhiddenlayers-1 then\n',nbhiddenlayers); 
fprintf (fid,'  call get_hiddenlayer_data (ilayer,conv2d_current_interior_bias,conv2d_current_interior_kernel)\n'); 
fprintf (fid,'  interior_input_padded(2:nlev+1,:) = new_state(:,:)\n');
fprintf (fid,'  interior_input_padded(1,:) = new_state(1,:)\n');
fprintf (fid,'  interior_input_padded(nlev+2,:) = input(nlev,:)\n');
fprintf (fid,'! ---- Go from 23x32 padded input to current dense layer to 21x32 for new_state that will arrive at next dense layer\n'); 
fprintf (fid,'  new_state(:,:) = 0. ! nlev x nbneurons \n');
fprintf (fid,'  do jnewchan = 1,nbneurons then\n'); % New channels
fprintf (fid,'    do kchan = 1,nbneurons then\n'); 
fprintf (fid,'      do k = 2,nlev+1 then\n')
fprintf (fid,'        do kfilt = 1,3 then\n')
fprintf (fid,'          new_state(k-1,jnewchan) = new_state(k-1,jnewchan) + interior_input_padded(k+kfilt-2,kchan)*conv2d_current_interior_kernel(kfilt,kchan,jnewchan)\n'); 
fprintf (fid,'        end do\n');
fprintf (fid,'      end do\n');
fprintf (fid,'    end do !kchan\n');
fprintf (fid,'! --- Apply bias and then leaky relu activation function:\n'); 
fprintf (fid,'    do k=1,nlev then\n'); 
fprintf (fid,'      new_state(k,jnewchan) = new_state(k,jnewchan) + conv2d_current_interior_bias(jnewchan)\n'); 
fprintf (fid,'      if (new_state(k,jnewchan) .lt. 0.) then\n'); 
fprintf (fid,'        new_state(k,jnewchan) = new_state(k,jnewchan)*0.3\n'); 
fprintf (fid,'      end if\n'); 
fprintf (fid,'    end do\n');
fprintf (fid,'  end do !jnewchan \n');
fprintf (fid,'end do !ilayer \n');

fprintf (fid,'\n\n! ---- Finally, for the outer layer, we go from 21x32 --> (no padding) --> (net matrices) --> the outer (2x21) --> its bias, NO activation :\n'); 
fprintf (fid,'call subroutine get_outerlayer_data (conv2d_outer_bias,conv2d_outer_kernel)\n'); 
fprintf (fid,'out_state(:,:) = 0. ! nlev x nchannel_out \n');
fprintf (fid,'do jnewchan = 1,nchannel_out then\n'); % New channels
fprintf (fid,'  do kchan = 1,nbneurons then\n'); 
fprintf (fid,'    do k=1,nlev then\n')
fprintf (fid,'      out_state(k,jnewchan) = out_state(k,jnewchan) + new_state(k,kchan)*conv2d_outer_kernel(kchan,jnewchan)\n'); 
fprintf (fid,'    end do\n');
fprintf (fid,'  end do !kchan\n');
fprintf (fid,'! --- Apply bias but NO activation function:\n'); 
fprintf (fid,'  do k=1,nlev then\n',nlev); 
fprintf (fid,'    out_state(k,jnewchan) = out_state(k,jnewchan) + conv2d_outer_bias(jnewchan)\n'); 
fprintf (fid,'  end do\n')
fprintf (fid,'end do ! jnewchan\n'); 
fprintf (fid,'\n\n! links to subroutine outputs: \n'); 
for kchan = 1:nchannel_out
    varnameout = char(output_varnames(kchan)); 
    fprintf (fid, '%s(:) = 0.\n',varnameout); 
    fprintf (fid, '%s((pver-nlev+1):pver) = out_state(:,%d)\n',varnameout,kchan); 
end
fprintf (fid, 'end subroutine cloudbrain_convo\n'); 

% ------- input layer definitions -------
fprintf (fid,'\n\nsubroutine get_inputlayer_data (bias,kernel)\n'); 
fprintf (fid,'real(r8), intent(out) :: kernel(3,nchannel_in,nlev)\n'); 
fprintf (fid,'real(r8), intent(out) :: bias(nbneurons)\n'); 
% Define the matrices for the first layer:
load conv2d_1_bias.mat
load conv2d_1_kernel.mat % 3 x 7 x 32
fprintf (fid,'bias(:) = %s\n',vec2f90str(conv2d_1_bias(:))); 
for kfilt = 1:3 % hardwired 3 pt filter depth. 
    for kchannel = 1:nchannel_in
        fprintf (fid,'kernel(%d,%d,:) = %s\n',kfilt,kchannel,vec2f90str(squeeze(conv2d_1_kernel(kfilt,kchannel,:)))); 
    end
end
fprintf (fid,'end subroutine get_inputlayer_data\n'); 

% --------- interior layer data definitions -----
fprintf (fid,'\n\nsubroutine get_hiddenlayer_data (ihiddenlayer,conv2d_current_interior_bias,conv2d_current_interior_kernel)\n'); 
fprintf (fid,'integer, intent(in) :: ihiddenlayer \n');
fprintf (fid,'real(r8), intent(out) :: conv2d_current_interior_bias(nbneurons)\n');
fprintf (fid,'real(r8), intent(out) :: conv2d_current_interior_kernel(3,nbneurons,nlev)\n'); 

fprintf (fid,'! -- define each layers bias and kernel matrices---\n'); 
for ilayer = 2:(nbhiddenlayers) % INSERT strip last to specific outer layer subroutine, different size.
    eval (sprintf ('load conv2d_%d_bias.mat',ilayer)); 
    eval (sprintf ('load conv2d_%d_kernel.mat',ilayer)); 
    eval (sprintf ('conv2d_current_interior_bias = conv2d_%d_bias',ilayer)); 
    eval (sprintf ('conv2d_current_interior_kernel = conv2d_%d_kernel',ilayer)); 
    fprintf (fid,'  if (ihiddenlayer .eq. %d) then\n',ilayer-1); 
    fprintf (fid,'    conv2d_current_interior_bias(:) = %s\n',vec2f90str(conv2d_current_interior_bias(:)));     
    for kfilt = 1:3 % hardwired 3 pt filter depth. 
      for kchannel = 1:nbneurons
        if (ilayer < nbhiddenlayers+1) % output has different size.
            fprintf (fid,'    conv2d_current_interior_kernel(%d,%d,:) = %s\n',kfilt,kchannel,vec2f90str(squeeze(conv2d_current_interior_kernel(kfilt,kchannel,:)))); 
        end
      end
    end
    fprintf (fid,'  end if ! def hidden layer %d\n',ilayer-1); 
end
fprintf (fid,'end subroutine get_hiddenlayer_data\n'); 

fprintf (fid,'\n\nsubroutine get_outerlayer_data (conv2d_outer_bias,conv2d_outer_kernel)\n'); 
fprintf (fid,'real(r8), intent(out) :: conv2d_outer_kernel (nbneurons,nchannel_out)\n'); 
fprintf (fid,'real(r8), intent(out) :: conv2d_outer_bias (nchannel_out)\n');  

eval (sprintf ('load conv2d_%d_bias.mat',nbhiddenlayers+1)); 
eval (sprintf ('outbias = conv2d_%d_bias',nbhiddenlayers+1)); 
eval (sprintf ('load conv2d_%d_kernel.mat',nbhiddenlayers+1)); 
eval (sprintf ('outkernel = conv2d_%d_kernel',nbhiddenlayers+1)); 
fprintf (fid,'conv2d_outer_bias(:) = %s\n',vec2f90str(outbias));
for kchannel=1:nchannel_out
   fprintf (fid,'conv2d_outer_kernel(:,%d) = %s\n',kchannel,vec2f90str(outkernel(kchannel,:))); 
end
fprintf (fid,'end subroutine get_outerlayer_data \n'); 

   fprintf (fid,'\n\nend module cloudbrain_convo_module\n'); 
 
    fclose (fid); 

error ('stop');
% LEFT OFF here. WARNING need to sanity check above against diffs added in
% Galen's v2 of convo_pritch_sanity!



    
% % INSERT left off here. 
% for kk=1:length(output_varnames)
%             fprintf (fid,'! ------- neural net calculation for %s ----\n',char(output_varnames(kk))); 
%             fprintf (fid,'call define_neuralnet_%s (weights_input,bias_input,bias_output,weights_output)\n',char(output_varnames(kk))); 
%             fprintf (fid,'h(:) = bias_input(:)\n'); 
%             fprintf (fid,'do i = 1,hiddenlayerSize\n'); 
%             fprintf (fid,'  do j=1,nlev\n'); 
%             fprintf (fid,'    h(i) = h(i) + weights_input(i,j)*input(j)\n'); 
%             fprintf (fid,'  end do\n'); 
%             fprintf (fid,'  h(i) = sigmoid(h(i))\n'); 
%             fprintf (fid,'end do\n'); 
%             fprintf (fid,'%s(:) = 0. ! output\n',char(output_varnames(kk))); 
%             fprintf (fid,'do i = 1,outputlayerSize\n'); 
%             fprintf (fid,'  %s(i) = %s(i) + bias_output(i)\n',char(output_varnames(kk)),char(output_varnames(kk))); 
%             fprintf (fid,'  do j = 1,hiddenlayerSize\n'); 
%             fprintf (fid,'    %s(i) = %s(i) + weights_output(i,j)*h(j)\n',char(output_varnames(kk)),char(output_varnames(kk))); 
%             fprintf (fid,'  end do\n'); 
%             fprintf (fid,'end do\n\n'); 
% 
%         end
%         
%         fprintf (fid, 'end subroutine cloudbrain\n\n\n'); 
%         
%         fprintf (fid, 'subroutine define_neuralnet_normalization (mu_in,std_in)\n'); 
%         fprintf (fid,'real(r8), intent(out) :: mu_in(nlev)\n'); 
%         fprintf (fid,'real(r8), intent(out) :: std_in(nlev)\n');
%         str = 'mu_in(:) = (/'; 
%         for k=1:nlev
%             str = [str sprintf('%10e, ',mymu(k))];
%         end
%         str = [str(1:end-2) ' /)\n'];
%         fprintf (fid,str);
% 
%         str = 'std_in(:) = (/'; 
%         for k=1:nlev
%             str = [str sprintf('%10e, ',mysig(k))];
%         end
%         str = [str(1:end-2) ' /)\n'];
%         fprintf (fid,str);     
%         fprintf (fid, 'end subroutine define_neuralnet_normalization\n\n'); 
%         
%         fprintf (fid,'function sigmoid(x) result(fx)\n');
%         fprintf (fid,'  real(r8), intent(in) :: x\n');
%         fprintf (fid,'  real(r8) :: fx\n')
%         fprintf (fid,'  fx = 1. / (1. + exp(-x))\n');
%         fprintf (fid,'end function sigmoid\n\n'); 
%     end
% 
%     % For each target, write the subroutine defining its neural net as read
%     % from the matlab data file in memory.
%     
% 
%     
%     fprintf (fid,'subroutine define_neuralnet_%s (weights_input,bias_input,bias_output,weights_output)\n',target_name); 
%     fprintf (fid,'real(r8), intent(out) :: bias_input(hiddenlayerSize)\n'); 
%     fprintf (fid,'real(r8), intent(out) :: bias_output(outputlayerSize)\n'); 
%     fprintf (fid,'real(r8), intent(out) :: weights_input(hiddenlayerSize,nlev)\n'); 
%     fprintf (fid,'real(r8), intent(out) :: weights_output(outputlayerSize,hiddenlayerSize)\n\n');
%         
%     str = 'bias_input(:) = (/';
%     for k=1:hiddenlayerSize
%         str = [str sprintf('%10e, ',bias_input(k))]; 
%     end
%     str = [str(1:end-2) ' /)\n'];
%     fprintf (fid,str);     
%     
%     str = 'bias_output(:) = (/';
%     for k=1:outputlayerSize        
%         str = [str sprintf('%10e, ',bias_output(k))]; 
%     end
%     str = [str(1:end-2) ' /)\n'];
%     fprintf (fid,str); 
%    
%     for kneuron=1:hiddenlayerSize
%         str = sprintf ('weights_input(%d,:) = (/',kneuron); 
%         for k=1:nlev
%             str = [str sprintf('%10e, ',weights_input(kneuron,k))];
%         end
%         str = [str(1:end-2) ' /)\n'];
%         fprintf (fid,str); 
%     end
%     
%     for kout=1:outputlayerSize
%         str = sprintf ('weights_output(%d,:) = (/',kout); 
%         for k=1:hiddenlayerSize
%             str = [str sprintf('%10e, ',weights_output(kout,k))];
%         end
%         str = [str(1:end-2) ' /)\n'];
%         fprintf (fid,str); 
%     end
% 
%     fprintf (fid,'\n\n');
%     fprintf (fid,'end subroutine define_neuralnet_%s \n\n\n',target_name); 
% 
% end
% 
% fprintf (fid,'end module cloudbrain_module\n'); 
% fclose (fid);
