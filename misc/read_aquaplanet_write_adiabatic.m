% read aquaplanet runs and define vector of tendencies due to convection
clear all;
close all;
clc;

  

% {'FLUT';'LHFLX';'P0';'PHQ';'PRECT';'PS';'QAP';'QRL';'QRS';'SHFLX';'SOLIN';'SPDQ';'SPDT';'TAP';'TPHYSTND'}
% TBP,QBP,PS,SOLIN,SHFLX,LHFLX,dTdt_adiabatic,dQdt_adiabatic

flag = 0;
% FOLDER = '../gentine_aqua_3';
% FOLDER = 'Pritchard_Aquaplanet';
FOLDER = '/beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/'
list_files = dir(FOLDER);
Nfiles  = 0;
for i=1:length(list_files)
    filename = list_files(i).name
    if(length(filename)>12)
        Nfiles = Nfiles +1 ;
    end
end
     
counter = 0;
for i=1:length(list_files)
    filename = list_files(i).name
    if(length(filename)>12)
        file = [FOLDER '/' filename];
        info = ncinfo(file);
        
        lon             = ncread(file,'lon');
        lat             = ncread(file,'lat');
        lev             = ncread(file,'lev');
        time            = ncread(file,'time');
        
        QAP_tmp         = ncread(file,'QAP');
        TAP_tmp         = ncread(file,'TAP');
        TPHYSTND_tmp    = ncread(file,'TPHYSTND');
        PHQ_tmp         = ncread(file,'PHQ');
        QRS             = ncread(file,'QRS');
        QRL             = ncread(file,'QRL');
        TBP             = TAP_tmp - TPHYSTND_tmp/1800.;
        QBP             = QAP_tmp - PHQ_tmp/1800.;
        
        dTdt_adiabatic  = zeros(size(TAP_tmp));
        dQdt_adiabatic  = zeros(size(TAP_tmp));
        dTdt_adiabatic(:,:,:,1) = 0.; % only for very first time step
        dQdt_adiabatic(:,:,:,1) = 0.;
        dTdt_adiabatic(:,:,:,2:end) = (TAP_tmp(:,:,:,2:end) - TAP_tmp(:,:,:,1:end-1))/1800. - TPHYSTND_tmp(:,:,:,2:end);
        dQdt_adiabatic(:,:,:,2:end) = (QAP_tmp(:,:,:,2:end) - QAP_tmp(:,:,:,1:end-1))/1800. - PHQ_tmp(:,:,:,2:end);
        if(counter>0)
            filename_previous = list_files(i-1).name;
            filename_previous = [FOLDER '/' filename_previous];
            QAP_previous    = ncread(filename_previous,'QAP');
            TAP_previous    = ncread(filename_previous,'TAP');
            dTdt_adiabatic(:,:,:,1) = (TAP_tmp(:,:,:,1) - TAP_previous(:,:,:,end))/1800. - TPHYSTND_tmp(:,:,:,1);
            dQdt_adiabatic(:,:,:,1) = (QAP_tmp(:,:,:,1) - QAP_previous(:,:,:,end))/1800. - PHQ_tmp(:,:,:,1);
        end
        
        
         
        % write those tendencies
%         dQdt_adiabatic(it) = (QAP(it) - QAP(it-1))/1800. - PTTEND(it)
%         dTdt_adiabattic (it) = (TAP(it)-TAP(it-1))/1800. - TPHYSTND(it) 
        nccreate(file,'dTdt_adiabatic',...
                     'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
                     'Datatype','single');
        nccreate(file,'dQdt_adiabatic',... 
                     'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
                     'Datatype','single');
        ncwrite(file,'dTdt_adiabatic',single(dTdt_adiabatic)  );
        ncwrite(file,'dQdt_adiabatic',single(dQdt_adiabatic)  );
        
        % now add previous time steps
        nccreate(file,'TBP',...
                     'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
                     'Datatype','single');
        nccreate(file,'QBP',... 
                     'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
                     'Datatype','single');
        ncwrite(file,'TBP',single(TBP)  );
        ncwrite(file,'QBP',single(QBP)  );
       
        % full physics
%         nccreate(file,'TPHYSTND',...
%                      'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
%                      'Datatype','single');
%         nccreate(file,'PHQ',... 
%                      'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
%                      'Datatype','single');
        nccreate(file,'TPHYSTND_NORAD',...
                     'Dimensions',{'lon',length(lon),'lat',length(lat),'lev', length(lev), 'time',length(time)},...
                     'Datatype','single');
        ncwrite(file,'TPHYSTND_NORAD',single(TPHYSTND_tmp-QRS-QRL)  );
%         ncwrite(file,'PHQ',single(QBP)  );
        
        counter = counter + 1; 
    end
end





'end'
