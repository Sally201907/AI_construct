load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin
;Observation
nf = addfile("fire_obs_2003_2022_GFED5.nc","r")
fire_obs = new((/8,228/), "float")
fire_obs(0,:) = nf->fires_sum(0:19*12-1)
fire_obs(1,:) = nf->fires_Europe(0:19*12-1)
fire_obs(2,:) = nf->fires_NAsia(0:19*12-1)
fire_obs(3,:) = nf->fires_ESAsia(0:19*12-1)
fire_obs(4,:) = nf->fires_SA(0:19*12-1)
fire_obs(5,:) = nf->fires_NA(0:19*12-1)
fire_obs(6,:) = nf->fires_Africa(0:19*12-1)
fire_obs(7,:) = nf->fires_Australia(0:19*12-1)


;compare model and AI-constrained in ssp585
model = (/"CESM2-WACCM","CESM2","CMCC-ESM2","CNRM-ESM2-1","E3SM-1-1","EC-Earth3-CC","EC-Earth3-Veg-LR",\
         "EC-Earth3-Veg","GFDL-ESM4","MPI-ESM1-2-LR","MRI-ESM2-0","NorESM2-LM"/)
nmodel = dimsizes(model)
print(nmodel)

tf = addfile("fire_CMIP6_2003_2100.nc","r")

nt = 2100-2003+1
cfire_model_3d = new((/8,nmodel,nt,12/),"float")
cfire_model_3d(0,:,:,:) = reshape(tf->cfire_model_global,(/nmodel,nt,12/))
cfire_model_3d(1,:,:,:) = reshape(tf->cfire_model_Europe,(/nmodel,nt,12/))
cfire_model_3d(2,:,:,:) = reshape(tf->cfire_model_NAsia,(/nmodel,nt,12/))
cfire_model_3d(3,:,:,:) = reshape(tf->cfire_model_ESAsia,(/nmodel,nt,12/))
cfire_model_3d(4,:,:,:) = reshape(tf->cfire_model_SA,(/nmodel,nt,12/))
cfire_model_3d(5,:,:,:) = reshape(tf->cfire_model_NA,(/nmodel,nt,12/))
cfire_model_3d(6,:,:,:) = reshape(tf->cfire_model_Africa,(/nmodel,nt,12/))
cfire_model_3d(7,:,:,:) = reshape(tf->cfire_model_Australia,(/nmodel,nt,12/))   

plotLabel = (/"~F22~a","~F22~b","~F22~c","~F22~d","~F22~e","~F22~f",\
             "~F22~g","~F22~h","~F22~i","~F22~j","~F22~k","~F22~l"/)
RegionLabel = (/"Global","Europe","NAsia","ESAsia","South America","North America","Africa","Aus+EQAsia"/)

;==============================================================================
fire_AI = new((/8,5,28,1176/), "float")
nf = addfile("./GFED5/output/SSP585/UMD_LUchange_SSP585_global.nc","r")
i=0
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)

nf = addfile("./GFED5/output/SSP585/UMD_LUchange_Europe.nc","r")
i=1
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_NAsia.nc","r")
i=2
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_ESAsia.nc","r")
i=3
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)



nf = addfile("./GFED5/output/SSP585/UMD_LUchange_SA.nc","r")
i=4
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_NA.nc","r")
i=5
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)

nf = addfile("./GFED5/output/SSP585/UMD_LUchange_Africa.nc","r")
i=6
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_Australia_EQAsia.nc","r")
i=7
fire_AI(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)

nt = 2100-2003+1
fire_AI_3d = new((/8,28,nt,12/), "float")
do i = 0,7
fire_AI_3d(i,:,:,:) = reshape(fire_AI(i,0,:,:),(/28,nt,12/))
end do 


;==================================================================
model = (/"CESM2-WACCM","CESM2","CMCC-ESM2","CNRM-ESM2-1","E3SM-1-1","EC-Earth3-CC","EC-Earth3-Veg-LR",\
         "EC-Earth3-Veg","GFDL-ESM4","MPI-ESM1-2-LR","MRI-ESM2-0","NorESM2-LM"/)
;remove "EC-Earth3-CC","EC-Earth3-Veg","EC-Earth3-Veg-LR","MRI-ESM2-0"
model_ind = (/0,1,2,3,4,8,9,11/)
print(model(model_ind))
; CALCULATE MIN & MAX
  mnmx      = new ((/8,7,12/), float)
  do i = 0,7
  mnmx(i,0,:) = dim_min_n(dim_avg_n(cfire_model_3d(i,model_ind,0:18,:),1),0)
  mnmx(i,1,:) = dim_max_n(dim_avg_n(cfire_model_3d(i,model_ind,0:18,:),1),0)
  mnmx(i,2,:) = dim_avg_n(dim_avg_n(cfire_model_3d(i,model_ind,0:18,:),1),0)
  mnmx(i,3,:) = dim_min_n(dim_avg_n(fire_AI_3d(i,:,0:18,:),1),0)
  mnmx(i,4,:) = dim_max_n(dim_avg_n(fire_AI_3d(i,:,0:18,:),1),0)
  mnmx(i,5,:) = dim_avg_n(dim_avg_n(fire_AI_3d(i,:,0:18,:),1),0)
  mnmx(i,6,:) = dim_avg_n(reshape(fire_obs(i,:),(/19,12/)),0)
  end do 


 mnmx!0= "Region"
 mnmx!1= "select"
 mnmx!2= "month"

cfire_model_annual = dim_sum_n(cfire_model_3d(:,model_ind,:,:),3)
;========================================================
   wks = gsn_open_wks("pdf","fig3")
plot = new((/9,3/),graphic)

     res                      = True           ; No plot options set.
     res@gsnDraw              = False          ; Do not draw plot
     res@gsnFrame             = False           ; Do not advance frame

res@tmXBLabelJust          = "TopLeft"
res@tmXBLabelAngleF        = -90
            res@tmXBLabelFontHeightF          = 0.025         ; change maj lat tm spacing
            res@tmYLLabelFontHeightF          = 0.025
            res@gsnLeftStringFontHeightF      = 0.025 
            res@gsnRightStringFontHeightF      = 0.025
            res@gsnCenterStringFontHeightF      = 0.025
     res@vpHeightF            = 0.3
     res@vpWidthF             = 0.395
res@trYMaxF            = 850.0
res@trYMinF            = 0.0
res@xyLineColors  := (/"blue","red","black"/) 
     res@xyDashPatterns       := (/0,0,0/)                 ; dashed line for 2nd)
     res@xyLineThicknesses    := (/3,3,3/)
time := ispan(1,12,1)
res@gsnLeftString = plotLabel(0)

res@lgLabelFontHeightF     = 0.0235                   ; change font height
res@xyExplicitLegendLabels := (/" CMIP6"," AI-constructed"," Observation"/)
res@tmXBLabelsOn =False
res@tiXAxisString          = ""
res@tiYAxisFontHeightF = 0.025
i=0
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@gsnCenterString = "Global"
res@pmLegendDisplayMode    = "Never"
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=1
res@tiYAxisString          =  "" 
res@tiYAxisString          =  "" 
            res@pmLegendWidthF         = 0.05                  ; Change width and
            res@pmLegendHeightF        = 0.13                  ; height of legend.
            res@pmLegendParallelPosF   = 0.35                  ; move units right
            res@pmLegendOrthogonalPosF = -0.63                  ; more neg = down
res@trYMaxF            = 100.0
res@trYMinF            = 0.0 
res@tiYAxisString          =  ""
res@gsnCenterString = "Europe"
res@gsnLeftString = plotLabel(i)  
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=2


res@trYMaxF            = 90.0
res@trYMinF            = 0.0 
res@gsnLeftString = plotLabel(i)  
res@gsnCenterStringOrthogonalPosF   = 0.05
res@gsnCenterString = "NAsia"
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=3
res@tiYAxisString          =  "" 
res@pmLegendDisplayMode    = "Never"
res@trYMaxF            = 200.0
res@trYMinF            = 0.0 
res@gsnLeftString = plotLabel(i)  
res@gsnCenterString = "ESAsia"
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=4
res@tiYAxisString          =  "" 
res@tiYAxisString          =  "" 
res@pmLegendDisplayMode    = "Never"
res@trYMaxF            = 350.0
res@trYMinF            = 0.0 
res@gsnLeftString = plotLabel(i)  

res@gsnCenterString = RegionLabel(i)
res@gsnCenterStringOrthogonalPosF   = 0.0
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=5
res@pmLegendDisplayMode    = "Never"
res@trYMaxF            = 150.0
res@trYMinF            = 0.0 
res@gsnLeftString = plotLabel(i)  
res@gsnCenterString = RegionLabel(i)
res@gsnCenterStringOrthogonalPosF   = 0.022
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=6
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@tmXBLabelsOn =True
  res@vpXF = 0.1
  res@vpYF = 0.38
res@trYMaxF            = 350.0
res@trYMinF            = 0.0 
res@gsnLeftString = plotLabel(i) 
res@gsnCenterStringOrthogonalPosF   = 0.0 
res@gsnCenterString = "Africa"
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

i=7
res@tiYAxisString          =  "" 
res@pmLegendDisplayMode    = "Always"
res@trYMaxF            = 180.0
res@trYMinF            = 0.0 
res@gsnLeftString = plotLabel(i)  
res@gsnCenterString = "Aus+EQAsia"
res@gsnCenterStringOrthogonalPosF   = 0.05
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,(/2,5,6/),:),res) 

; Create a plot with the area between both curves filled in blue.

res@tiYAxisString          =  "" 
res@tmXBLabelsOn =False
  res@gsnLeftString = ""
  res@pmLegendDisplayMode    = "Never"
  delete(res@xyLineColors)
  res@gsnXYFillColors = "lightblue"
  res@xyLineColor     = -1                           ; We don't want the line, so make it transparent.
res@trYMaxF            = 850.0
res@trYMinF            = 0.0 
  res@gsnCenterString = ""               ; We don't want the line, so make it transparent.
  res@gsnXYFillColors = "lightblue"
  res@gsnXYFillOpacities = 0.5

  plot(0,1)  = gsn_csm_xy (wks,time,mnmx(0,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(0,2) = gsn_csm_xy (wks,time,mnmx(0,3:4,:),res)  ; Create another filled XY plot.

;Europe
res@tiYAxisString          =  "" 
res@tiYAxisString          =  "" 
res@trYMaxF            = 100.0
res@trYMinF            = 0.0                     ; We don't want the line, so make it transparent.
  res@gsnXYFillColors = "lightblue"
  plot(1,1)  = gsn_csm_xy (wks,time,mnmx(1,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(1,2) = gsn_csm_xy (wks,time,mnmx(1,3:4,:),res)  ; Create another filled XY plot.

;NAsia
res@trYMaxF            = 90.0
res@trYMinF            = 0.0                   ; We don't want the line, so make it transparent.
  res@gsnXYFillColors = "lightblue"
  plot(2,1)  = gsn_csm_xy (wks,time,mnmx(2,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(2,2) = gsn_csm_xy (wks,time,mnmx(2,3:4,:),res)  ; Create another filled XY plot.

;ESAsia
res@trYMaxF            = 200.0
res@trYMinF            = 0.0 
  res@gsnXYFillColors = "lightblue"
  plot(3,1)  = gsn_csm_xy (wks,time,mnmx(3,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(3,2) = gsn_csm_xy (wks,time,mnmx(3,3:4,:),res)  ; Create another filled XY plot.


;SA
res@tiYAxisString          =  "" 
i=4
res@tiYAxisString          =  "" 
res@trYMaxF            = 350.0
res@trYMinF            = 0.0 
  res@gsnXYFillColors = "lightblue"
  plot(i,1)  = gsn_csm_xy (wks,time,mnmx(i,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(i,2) = gsn_csm_xy (wks,time,mnmx(i,3:4,:),res)  ; Create another filled XY plot.

;NA
i=5
res@trYMaxF            = 150.0
res@trYMinF            = 0.0
  res@gsnXYFillColors = "lightblue"
  plot(i,1)  = gsn_csm_xy (wks,time,mnmx(i,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(i,2) = gsn_csm_xy (wks,time,mnmx(i,3:4,:),res)  ; Create another filled XY plot.

;Africa
res@tmXBLabelsOn =True
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
i=6
res@tmXBLabelsOn =True
res@trYMaxF            = 350.0
res@trYMinF            = 0.0  
  res@gsnXYFillColors = "lightblue"
  plot(i,1)  = gsn_csm_xy (wks,time,mnmx(i,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(i,2) = gsn_csm_xy (wks,time,mnmx(i,3:4,:),res)  ; Create another filled XY plot.

;Australia
i=7
res@tiYAxisString          =  "" 
res@trYMaxF            = 180.0
res@trYMinF            = 0.0 
  res@gsnXYFillColors = "lightblue"
  plot(i,1)  = gsn_csm_xy (wks,time,mnmx(i,0:1,:),res)  ; Create filled XY plot.
  res@gsnXYFillColors = "LightPink"
  plot(i,2) = gsn_csm_xy (wks,time,mnmx(i,3:4,:),res)  ; Create another filled XY plot.

do i = 0,7
    overlay(plot(i,1),plot(i,2))
    overlay(plot(i,1),plot(i,0))
end do 
;the future change of global fire emissions
delete(mnmx)
delete(time)

;boxplot:evaluate the long-term trend

rc_model = new((/8,8/), "float")
alpha_model = new((/8,8/), "float")
   time = ispan(2003, 2021, 1)
   rc_model = regCoef_n(time, dim_sum_n_Wrap(cfire_model_3d(:,model_ind,0:18,:), 3), 0, 2)  
   tval= onedtond(rc_model@tval , dimsizes(rc_model))
   df  = onedtond(rc_model@nptxy, dimsizes(rc_model)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_model = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_model)
   delete(tval)
   delete(df)
   delete(b)

   rc_AI = new((/8,28/), "float")
   alpha_AI = new((/8,28/), "float")
   time = ispan(2003, 2021, 1)
   rc_AI = regCoef_n(time, dim_sum_n_Wrap(fire_AI_3d(:,:,0:18,:),3), 0, 2)  
   tval= onedtond(rc_AI@tval , dimsizes(rc_AI))
   df  = onedtond(rc_AI@nptxy, dimsizes(rc_AI)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_AI = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_AI)
   delete(tval)
   delete(df)
   delete(b)

   rc_obs = new((/8/), "float")
   alpha_obs = new((/8/), "float")
   time = ispan(2003, 2021, 1)
   fire_obs_3d = reshape(fire_obs,(/8,19,12/))
   rc_obs = regCoef_n(time, dim_sum_n_Wrap(fire_obs_3d(:,0:18,:),2), 0, 1)  
   tval= onedtond(rc_obs@tval , dimsizes(rc_obs))
   df  = onedtond(rc_obs@nptxy, dimsizes(rc_obs)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_obs = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_obs)
   delete(tval)
   delete(df)
   delete(b)

  ip = dim_pqsort_n(rc_model, 2, 1) 
  delete(ip)
  ip = dim_pqsort_n(rc_AI, 2, 1)
  delete(ip)

  iarr=new((/16,5/),"float") ; fill with minimum, 25th percentile, median, 
  dimt = 9
  x25  = round(.25*dimt,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimt,3)-1     ; at 0 
  do i = 0,7     
     iarr(2*i,:) = (/min(rc_model(i,:)),rc_model(i,x25),dim_median(rc_model(i,:)),rc_model(i,x75),max(rc_model(i,:))/) 
  end do 
  dimt = 28
  x25  = round(.25*dimt,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimt,3)-1  
  do i = 0,7    
     iarr(2*i+1,:) = (/min(rc_AI(i,:)),rc_AI(i,x25),dim_median(rc_AI(i,:)),rc_AI(i,x75),max(rc_AI(i,:))/) 
  end do 
  delete(rc_AI)

            res2 = True 
            res2@tiXAxisString          = ""; xaxis string
     res2@vpHeightF            = 0.35
     res2@vpWidthF             = 0.5
              ;res2@tmYLPrecision = 1
            ;===========set the boundary of the picture==============================
            res2@tmBorderThicknessF            = 0.5
            res2@tmXBMajorThicknessF           = 0.5
            res2@tmYLMajorThicknessF           = 0.5
            res2@tmXBMinorOn                   = False
            res2@tmXTOn                        = False
            res2@tmYROn                        = False

            res2@tmXBLabelFontHeightF          =0.018         ; change maj lat tm spacing
            res2@tmYLLabelFontHeightF          =0.018
res2@tmYRBorderOn    = True         ; Habilita a borda do eixo y direito (YR). 
res2@tmXTBorderOn    = True         ; Habilita a borda do eixo x superior (XB). 
;res2@tmXBMajorOutwardLengthF = 0.01
res2@tmXBMajorLengthF        = 0.00

res2@tmYLMajorOutwardLengthF = 0.01
res2@tmYLMajorLengthF        = 0.01
res2@tmYLMinorOutwardLengthF = 0.006
res2@tmYLMinorLengthF        = 0.006
  res2@trYMaxF    = 80.0
  res2@trYMinF    = -50.0
  res2@tiYAxisFontHeightF =0.018
  res2@tiXAxisFontHeightF =0.018
  res2@tiYAxisString          =  "Trend (Tg/yr)"
  res2@tiXAxisString          =  ""
  res2@tmXBLabelAngleF        = -90
  res2@tmXBLabelJust          = "TopRight"
  res2@trXMinF     = 1
  res2@trXMaxF     = 32
  res2@tmXBLabels  = (/"Global","","Europe","","NAsia","","ESAsia","",\
                      "South America","","North America","","Africa","","Aus+EQAsia",""/)  
            res2@tmXBLabelJust          = "TopLeft"
            res2@tmXBLabelDeltaF        = -0.5
;**********************************************
; resources for polylines that draws the boxes
;**********************************************  
  llres2                   = True      
  llres2@gsLineThicknessF  = 3.0                 ; line thickness 
;********************************************* *
; res2ources that control color and width of boxes
;**********************************************  
  opti          = True      
  opti@boxWidth = 0.5       ; Width of box (x units)
  opti@boxColors = (/"black","black","blue","blue","red","red","coral4","coral4",\
                    "forestgreen","forestgreen","skyblue","skyblue",\
                    "lightcoral","lightcoral","lightseagreen","lightseagreen"/) 
;***********************************************
  res2@tiXAxisString          =  ""
  time := (/2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31/)
  plot(8,1) = boxplot(wks,time,iarr(:,:),opti,res2,llres2)   

  mres               = True                     ; marker mods desired
  mres@gsMarkerColor = "black"              ; change color
  mres@gsMarkerIndex = 3                        ; change style
  mres@gsMarkerSizeF = 15.                      ; change size
  mres@gsMarkerThicknessF = 2.5
  dum = new((/8,2/),graphic)
  do i = 0,7
    dum(i,0) = gsn_add_polymarker(wks,plot(8,1),time(2*i),rc_obs(i),mres) 
    dum(i,1) = gsn_add_polymarker(wks,plot(8,1),time(2*i+1),rc_obs(i),mres) 
  end do 
  
  delete(iarr)
  resP                        = True                   ; modify the panel plot
  resP@gsnFrame               = False                ; don't advance panel plot
  resP@gsnPanelLabelBar       = False                  ; add common colorbar
  resP@gsnPanelXWhiteSpacePercent = 0.0
  resP@gsnPanelYWhiteSpacePercent = 0.0
 
  resP@gsnPanelMainString     = ""
  resP@gsnPanelXWhiteSpacePercent = 0.0
  resP@gsnPanelYWhiteSpacePercent = 0.1
  gsn_panel(wks, plot(0:7,1), (/3,3/), resP)
  frame(wks)
  draw(plot(8,1))
  tres2               = True                    ; text mods desired
  tres2@txFontHeightF = 0.018
  gsn_text_ndc(wks,"~F22~i",0.20,0.82, tres2)
  frame(wks)


end