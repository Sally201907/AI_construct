load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin

fire_AI_ssp585 = new((/8,5,28,1176/), "float")
nf = addfile("./GFED5/output/SSP585/UMD_LUchange_SSP585_global.nc","r")
i=0
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)

nf = addfile("./GFED5/output/SSP585/UMD_LUchange_Europe.nc","r")
i=1
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_NAsia.nc","r")
i=2
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_ESAsia.nc","r")
i=3
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)



nf = addfile("./GFED5/output/SSP585/UMD_LUchange_SA.nc","r")

i=4
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_NA.nc","r")
i=5
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)

nf = addfile("./GFED5/output/SSP585/UMD_LUchange_Africa.nc","r")
i=6
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nf = addfile("./GFED5/output/SSP585/UMD_LUchange_Australia_EQAsia.nc","r")
i=7
fire_AI_ssp585(i,0,:,:) = doubletofloat(nf->fires_Hist) ;model,time
fire_AI_ssp585(i,1,:,:) = doubletofloat(nf->fires_Hist_VPD)
fire_AI_ssp585(i,2,:,:) = doubletofloat(nf->fires_Hist_tas)
fire_AI_ssp585(i,3,:,:) = doubletofloat(nf->fires_Hist_forest)
fire_AI_ssp585(i,4,:,:) = doubletofloat(nf->fires_Hist_cropland)


nt = 2100-2003+1
fire_AI_ssp585_3d = new((/8,5,28,nt,12/), "float")
do i = 0,7
fire_AI_ssp585_3d(i,:,:,:,:) = reshape(fire_AI_ssp585(i,:,:,:),(/5,28,nt,12/))
end do 
fire_AI_ssp585_annual = dim_sum_n_Wrap(fire_AI_ssp585_3d, 4)
;================================================================================================
fire_AI_ssp245 = new((/8,23,1176/), "float")
nf = addfile("./GFED5/output/SSP245/UMD_LUchange_SSP245_global.nc","r")
i=0
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_Europe.nc","r")
i=1
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_NAsia.nc","r")
i=2
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_ESAsia.nc","r")
i=3
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_SA.nc","r")
i=4
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_NA.nc","r")
i=5
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_Africa.nc","r")
i=6
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

nf = addfile("./GFED5/output/SSP245/UMD_LUchange_Australia_EQAsia.nc","r")
i=7
fire_AI_ssp245(i,:,:) = doubletofloat(nf->fires_Hist) ;model,time

fire_AI_ssp245_annual = dim_sum_n_Wrap(reshape(fire_AI_ssp245,(/8,23,nt,12/)), 3)


   rc_AI_ssp585 = new((/8,28/), "float")
   alpha_AI_ssp585 = new((/8,28/), "float")
   time := ispan(2024, 2053, 1)
   rc_AI_ssp585 = regCoef_n(time, fire_AI_ssp585_annual(:,0,:,21:50), 0, 2)  
   tval= onedtond(rc_AI_ssp585@tval , dimsizes(rc_AI_ssp585))
   df  = onedtond(rc_AI_ssp585@nptxy, dimsizes(rc_AI_ssp585)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_AI_ssp585 = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_AI_ssp585)
   delete(tval)
   delete(df)
   delete(b)
  ip = dim_pqsort_n(rc_AI_ssp585, 2, 1) ;ascending order, sort x also
  delete(ip)


   rc_AI_ssp245 = new((/8,23/), "float")
   alpha_AI_ssp245 = new((/8,23/), "float")
   rc_AI_ssp245 = regCoef_n(time, fire_AI_ssp245_annual(:,:,21:50), 0, 2)  
   tval= onedtond(rc_AI_ssp245@tval , dimsizes(rc_AI_ssp245))
   df  = onedtond(rc_AI_ssp245@nptxy, dimsizes(rc_AI_ssp245)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_AI_ssp245 = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_AI_ssp245)
   delete(tval)
   delete(df)
   delete(b)
  ip = dim_pqsort_n(rc_AI_ssp245, 2, 1) ;ascending order, sort x also.dim=1
  delete(ip)


  iarr=new((/16,2,5/),"float") 
  i=0
  dimt = 28
  x25  = round(.25*dimt,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimt,3)-1     ; at 0 
  do j = 0,7    
     iarr(2*j,0,:) = (/min(rc_AI_ssp585(j,:)),rc_AI_ssp585(j,x25),dim_median(rc_AI_ssp585(j,:)),rc_AI_ssp585(j,x75),max(rc_AI_ssp585(j,:))/) 
  end do 

  dimt = 23
  x25  = round(.25*dimt,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimt,3)-1     ; at 0
  i=1
  do j = 0,7    
     iarr(2*j+1,0,:) = (/min(rc_AI_ssp245(j,:)),rc_AI_ssp245(j,x25),dim_median(rc_AI_ssp245(j,:)),rc_AI_ssp245(j,x75),max(rc_AI_ssp245(j,:))/) 
  end do    


   time = ispan(2071, 2100, 1)
   rc_AI_ssp585 = regCoef_n(time, fire_AI_ssp585_annual(:,0,:,68:), 0, 2)  
   tval= onedtond(rc_AI_ssp585@tval , dimsizes(rc_AI_ssp585))
   df  = onedtond(rc_AI_ssp585@nptxy, dimsizes(rc_AI_ssp585)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_AI_ssp585 = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_AI_ssp585)
   delete(tval)
   delete(df)
   delete(b)


   rc_AI_ssp245 = regCoef_n(time, fire_AI_ssp245_annual(:,:,68:), 0, 2)  
   tval= onedtond(rc_AI_ssp245@tval , dimsizes(rc_AI_ssp245))
   df  = onedtond(rc_AI_ssp245@nptxy, dimsizes(rc_AI_ssp245)) - 2
   b = tval    ; b must be same size as tval (and df)
   b = 0.5
   alpha_AI_ssp245 = betainc(df/(df+tval^2),df/2.0,b)       
   printVarSummary(rc_AI_ssp245)
   delete(tval)
   delete(df)
   delete(b)


  ip = dim_pqsort_n(rc_AI_ssp585, 2, 1) ;ascending order, sort x also
  delete(ip)
  ip = dim_pqsort_n(rc_AI_ssp245, 2, 1) ;ascending order, sort x also.dim=1
  delete(ip)

  i=0
  dimt = 28
  x25  = round(.25*dimt,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimt,3)-1     ; at 0 
  do j = 0,7    
     iarr(2*j,1,:) = (/min(rc_AI_ssp585(j,:)),rc_AI_ssp585(j,x25),dim_median(rc_AI_ssp585(j,:)),rc_AI_ssp585(j,x75),max(rc_AI_ssp585(j,:))/) 
  end do 

  dimt = 23
  x25  = round(.25*dimt,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimt,3)-1     ; at 0
  i=1
  do j = 0,7    
     iarr(2*j+1,1,:) = (/min(rc_AI_ssp245(j,:)),rc_AI_ssp245(j,x25),dim_median(rc_AI_ssp245(j,:)),rc_AI_ssp245(j,x75),max(rc_AI_ssp245(j,:))/) 
  end do    


plotLabel = (/"~F22~a","~F22~b","~F22~c","~F22~d","~F22~e","~F22~f",\
             "~F22~g","~F22~h","~F22~i","~F22~j","~F22~k","~F22~l"/)
RegionLabel = (/"Global","Europe","NAsia","ESAsia","South America","North America","Africa","Aus+EQAsia"/)
 
   wks = gsn_open_wks("pdf","fig5")

plot = new(2,graphic)  

            res = True 
            res@tiXAxisString          = ""; xaxis string
     res@vpHeightF            = 0.3
     res@vpWidthF             = 0.6
              ;res@tmYLPrecision = 1
            ;===========set the boundary of the picture==============================
            res@tmBorderThicknessF            = 0.5
            res@tmXBMajorThicknessF           = 0.5
            res@tmYLMajorThicknessF           = 0.5
            res@tmXBMinorOn                   = False
            res@tmXTOn                        = False
            res@tmYROn                        = False

            res@tmXBLabelFontHeightF          =0.018         ; change maj lat tm spacing
            res@tmYLLabelFontHeightF          =0.018
res@tmYRBorderOn    = True         ; Habilita a borda do eixo y direito (YR). 
res@tmXTBorderOn    = True         ; Habilita a borda do eixo x superior (XB). 
res@tmXBMajorOutwardLengthF = 0.00
res@tmXBMajorLengthF        = 0.00

res@tmYLMajorLengthF        = 0.01
res@tmYLMinorOutwardLengthF = 0.006
res@tmYLMinorLengthF        = 0.006

  res@trYMaxF    = 35.0
  res@trYMinF    = -10.0
  res@tiYAxisFontHeightF =0.018
  res@tiXAxisFontHeightF =0.018
  res@tiYAxisString          =  "Trend (Tg/yr)"
  res@tiXAxisString          =  ""
  res@tmXBLabelAngleF        = -90
  res@tmXBLabelJust          = "TopRight"
  res@trXMinF     = 1
  res@trXMaxF     = 32
  res@tmXBLabels  = (/"Global","","Europe","","NAsia","","ESAsia","",\
                      "South America","","North America","","Africa","","Aus+EQAsia",""/)  
            res@tmXBLabelJust          = "TopLeft"
            res@tmXBLabelDeltaF        = -0.5
;**********************************************
; resources for polylines that draws the boxes
;**********************************************  
  llres                   = True      
  llres@gsLineThicknessF  = 3.0                 ; line thickness 
;********************************************* *
; resources that control color and width of boxes
;**********************************************  
  opti          = True      
  opti@boxWidth = 0.5       ; Width of box (x units)
  opti@boxColors := (/"black","black","blue","blue","red","red","coral4","coral4",\
                    "forestgreen","forestgreen","skyblue","skyblue",\
                    "lightcoral","lightcoral","lightseagreen","lightseagreen"/) 
;***********************************************
  res@tiXAxisString          =  ""
  res@tmXBLabelsOn = False
  time := (/2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31/)
  plot(0) = boxplot(wks,time,iarr(:,0,:),opti,res,llres)   
  res@tmXBLabelsOn = True
  plot(1) = boxplot(wks,time,iarr(:,1,:),opti,res,llres)  
  delete(iarr)

  tres               = True                    ; text mods desired
  tres@txFontHeightF = 0.022
  gsn_text_ndc(wks,"~F22~a",0.20,0.979, tres)
  gsn_text_ndc(wks,"2024-2053",0.55,0.979, tres)
  gsn_text_ndc(wks,"~F22~b",0.20,0.58, tres)
  gsn_text_ndc(wks,"2071-2100",0.55,0.58, tres)
  resP                        = True                   ; modify the panel plot
  resP@gsnFrame               = False                 ; don't advance panel plot
  resP@gsnPanelLabelBar       = False                  ; add common colorbar
  resP@gsnPanelXWhiteSpacePercent = 0.0
  resP@gsnPanelYWhiteSpacePercent = 0.0
  resP@gsnPanelMainString     = ""
resP@gsnPanelYWhiteSpacePercent = 0.0
resP@gsnPanelBottom = 0.2
gsn_panel(wks, plot, (/2,1/), resP)
  frame(wks)

end 