load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin

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
cfire_model_annual = dim_sum_n(cfire_model_3d(:,model_ind,:,:),3)

  nt = 2100-2022+1 
  mnmx      = new ((/8,5,3,nt/), float)

  do i = 0,7

  mnmx(i,:,0,:) = dim_min_n(dim_sum_n(reshape(fire_AI(i,:,:,19*12:),(/5,28,nt,12/)),3),1)
  mnmx(i,:,1,:) = dim_max_n(dim_sum_n(reshape(fire_AI(i,:,:,19*12:),(/5,28,nt,12/)),3),1)
  mnmx(i,:,2,:) = dim_avg_n(dim_sum_n(reshape(fire_AI(i,:,:,19*12:),(/5,28,nt,12/)),3),1)

end do 


;========================================================
   wks = gsn_open_wks("pdf","fig4")
plot = new((/8,6/),graphic)
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

  res@gsnXYFillOpacities = 1.0
            res@tmXBLabelFontHeightF          = 0.028         ; change maj lat tm spacing
            res@tmYLLabelFontHeightF          = 0.028
            res@gsnLeftStringFontHeightF      = 0.028 
            res@gsnRightStringFontHeightF      = 0.028
            res@gsnCenterStringFontHeightF      = 0.028
            res@tiYAxisFontHeightF = 0.028
res@tmXBLabelJust          = "TopLeft"
 res@tmXBLabelAngleF        = -45
res@lgItemOrder                  := (/4,3,2,1,0/)
     res@xyDashPatterns       := (/0,0,0,0,0/)                 ; dashed line for 2nd)
     res@xyLineThicknesses    := (/2.5,2,2,2,2/)
res@trXMaxF            = 2100
res@trXMinF            = 2022
res@xyLineColors  := (/"black","blue","red","forestgreen","tan4"/) 
time = ispan(2022,2100,1)
res@vpHeightF= 0.4
res@vpWidthF = 0.6

  delete(res@tmXBValues)
  delete(res@tmXBLabels)
  res@tmXBValues             = ispan(2022,2100,10)
  res@tmXBLabels             = ispan(2022,2100,10)
res@lgLabelFontHeightF     = 0.0239                   ; change font height
res@xyExplicitLegendLabels := (/" Combined"," VPD"," Temperature", " Forest", " Cropland"/)
delete(res@gsnXYFillColors)
  res@trYMaxF            = 5500.0
  res@trYMinF            = 3100.0 
res@tmXBLabelsOn =False
i=0
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
            res@pmLegendParallelPosF   = 0.25                   ; move units right
            res@pmLegendOrthogonalPosF = -0.65                  ; more neg = down
            res@pmLegendWidthF         = 0.08                  ; Change width and
            res@pmLegendHeightF        = 0.20                  ; height of legend.
res@pmLegendDisplayMode    = "Always"
res@gsnCenterString = "Global"
res@gsnLeftString = plotLabel(i)
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)

i=1
res@pmLegendDisplayMode    = "Never"
res@tiYAxisString          =  "" 
res@gsnCenterString = "Europe"
  res@trYMaxF            = 180.0
  res@trYMinF            = 0.0


res@gsnLeftString = plotLabel(i)
res@gsnCenterStringOrthogonalPosF   = -0.01
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)

i=2
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@gsnCenterString = "NAsia"
res@trYMaxF            = 1300.0
res@trYMinF            = 0.0
res@gsnLeftString = plotLabel(i)
res@gsnCenterStringOrthogonalPosF   = 0.01
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)

i=3
res@tiYAxisString          =  "" 
res@gsnCenterString = "ESAsia"
  res@trYMaxF            = 1200.0
  res@trYMinF            = 100.0
res@gsnCenterStringOrthogonalPosF   = 0.0
res@gsnLeftString = plotLabel(i)
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)


i=4
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@pmLegendDisplayMode    = "Never"
res@gsnCenterString = RegionLabel(i)
res@trYMaxF            = 2000.0
res@trYMinF            = 100.0
res@gsnLeftString = plotLabel(i)
res@gsnCenterStringOrthogonalPosF   = 0.022
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)
i=5
res@tiYAxisString          =  "" 
res@gsnCenterString = RegionLabel(i)
res@gsnCenterStringOrthogonalPosF   = 0.01
res@trYMaxF            = 600.0
res@trYMinF            = 100.0
res@gsnLeftString = plotLabel(i)
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)

i=6
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@tmXBLabelsOn = True
res@gsnCenterStringOrthogonalPosF   = 0.0
res@gsnCenterString = "Africa"
res@trYMaxF            = 2100.0
res@trYMinF            = 1200.0
res@gsnLeftString = plotLabel(i)
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)

i=7
res@tiYAxisString          =  "" 
res@gsnCenterString = "Aus+EQAsia"
res@gsnCenterStringOrthogonalPosF   = 0.022
res@trYMaxF            = 1000.0
res@trYMinF            = 0.0
res@gsnLeftString = plotLabel(i)
plot(i,0) = gsn_csm_xy (wks,time,mnmx(i,:,2,:),res)

res@tiYAxisString          =  "Carbon Emissions (Tg)" 
  res@gsnLeftString = ""
  delete(res@xyLineColors)
  res@pmLegendDisplayMode    = "Never"
res@tmXBLabelsOn = False  
  res@xyLineColor     = -1
  res@gsnCenterString = ""  
  res@trYMaxF            = 5500.0
  res@trYMinF            = 3100.0  
  colors = (/"lightgrey","blue","LightPink","aquamarine3","tan"/)
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(0,i+1)  = gsn_csm_xy (wks,time,mnmx(0,i,0:1,:),res) 
  end do  

res@tiYAxisString          =  "" 
  res@trYMaxF            = 180.0
  res@trYMinF            = 0.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(1,i+1)  = gsn_csm_xy (wks,time,mnmx(1,i,0:1,:),res) 
  end do  

res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@trYMaxF            = 1300.0
res@trYMinF            = 0.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(2,i+1)  = gsn_csm_xy (wks,time,mnmx(2,i,0:1,:),res) 
  end do  

res@tiYAxisString          =  "" 
  res@trYMaxF            = 1200.0
  res@trYMinF            = 100.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(3,i+1)  = gsn_csm_xy (wks,time,mnmx(3,i,0:1,:),res) 

  end do  

res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@trYMaxF            = 2000.0
res@trYMinF            = 100.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(4,i+1)  = gsn_csm_xy (wks,time,mnmx(4,i,0:1,:),res) 
  end do  

res@tiYAxisString          =  "" 
res@trYMaxF            = 600.0
res@trYMinF            = 100.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(5,i+1)  = gsn_csm_xy (wks,time,mnmx(5,i,0:1,:),res) 
  end do  

res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@tmXBLabelsOn = True
res@trYMaxF            = 2100.0
res@trYMinF            = 1200.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(6,i+1)  = gsn_csm_xy (wks,time,mnmx(6,i,0:1,:),res) 
  end do  

res@tiYAxisString          =  "" 
res@trYMaxF            = 1000.0
res@trYMinF            = 0.0
  do i = 0,4
  res@gsnXYFillColors = colors(i)                     ; We don't want the line, so make it transparent.
  plot(7,i+1)  = gsn_csm_xy (wks,time,mnmx(7,i,0:1,:),res) 
  end do  

do i = 0,7
  overlay(plot(i,1),plot(i,0))
end do 
  resP                        = True                   ; modify the panel plot
  resP@gsnFrame               = False                ; don't advance panel plot
  resP@gsnPanelLabelBar       = False                  ; add common colorbar
  resP@gsnPanelXWhiteSpacePercent = 0.0
  resP@gsnPanelYWhiteSpacePercent = 0.0
 
  resP@gsnPanelMainString     = ""
  resP@gsnPanelXWhiteSpacePercent = 0.0
  resP@gsnPanelYWhiteSpacePercent = 0.1
resP@gsnPanelYWhiteSpacePercent = 0.0
resP@gsnPanelBottom = 0.1
gsn_panel(wks, plot(:,1), (/4,2/), resP)
frame(wks)

end