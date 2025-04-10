load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin
tf = addfile("fire_obs_2003_2022_GFED5.nc","r")

data = new((/8,6,19/), "float")
data(0,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_sum(0:19*12-1),(/19,12/)),1))
data(0,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_global,(/5,19,12/)),2))
data(1,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_Europe(0:19*12-1),(/19,12/)),1))
data(1,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_Europe,(/5,19,12/)),2))
data(2,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_NAsia(0:19*12-1),(/19,12/)),1))
data(2,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_NAsia,(/5,19,12/)),2))
data(3,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_ESAsia(0:19*12-1),(/19,12/)),1))
data(3,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_ESAsia,(/5,19,12/)),2))
data(4,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_SA(0:19*12-1),(/19,12/)),1))
data(4,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_SA,(/5,19,12/)),2))
data(5,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_NA(0:19*12-1),(/19,12/)),1))
data(5,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_NA,(/5,19,12/)),2))
data(6,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_Africa(0:19*12-1),(/19,12/)),1))
data(6,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_Africa,(/5,19,12/)),2))
data(7,0,:) = doubletofloat(dim_sum_n_Wrap(reshape(tf->fires_Australia(0:19*12-1),(/19,12/)),1))
data(7,1:5,:) = doubletofloat(dim_sum_n_Wrap(reshape(nf->fires_Australia,(/5,19,12/)),2))


   wks = gsn_open_wks("pdf","fig1")
     res                      = True           ; No plot options set.
     res@gsnDraw              = False          ; Do not draw plot
     res@gsnFrame             = False           ; Do not advance frame
     res@vpHeightF            = 0.4
     res@vpWidthF             = 0.6
res@tmXBLabelAngleF        = -30
            res@tiXAxisFontHeightF     = 0.035
            res@tiYAxisFontHeightF     = 0.035
            res@tmXBLabelFontHeightF          = 0.035         ; change maj lat tm spacing
            res@tmYLLabelFontHeightF          = 0.035 
            res@gsnLeftStringFontHeightF      = 0.035  
            res@gsnCenterStringFontHeightF      = 0.035
            res@pmLegendWidthF         = 0.09                  ; Change width and
            res@pmLegendHeightF        = 0.22                  ; height of legend.
            res@lgLabelFontHeightF     = 0.028                   ; change font height
            res@pmLegendParallelPosF   = 0.25                 ; move units right
            res@pmLegendOrthogonalPosF = -1.16                ; more neg = down
            res@tmXBLabelJust          = "TopLeft"
            res@tmXBLabelDeltaF        = -0.5
res@pmLegendDisplayMode    = "Never"
            ;===========set the boundary of the picture==============================
            res@tmBorderThicknessF            = 0.5
            res@tmXBMajorThicknessF           = 0.5
            res@tmYLMajorThicknessF           = 0.5
            ;res@tmYLMinorThicknessF           = 0.5
            res@tmXBMinorOn                   = False
            res@tmXTOn                        = False
            res@tmYROn                        = False
            res@tmXBLabelFontHeightF          = 0.018         ; change maj lat tm spacing
            res@tmYLLabelFontHeightF          = 0.018
            res@tmXBMajorLengthF              = 0.018          ; change the tickmark length 
            res@tmYLMajorLengthF              = 0.006          ; change the tickmark length
            res@tmYLMinorLengthF              = 0.003
            res@lgItemOrder                   = (/1,0/)
            res@gsnLeftStringFontHeightF      = 0.018        
            res@gsnLeftString   = "" 
            res@gsnLeftStringOrthogonalPosF   = 0.02
            res@gsnLeftStringParallelPosF     = -0.16
            res@gsnCenterStringOrthogonalPosF   = 0.01
            res@gsnCenterString = ""
            res@gsnCenterStringFontHeightF     = 0.018

res@gsnLeftString   = plotLabel(0)
res@gsnCenterString = "Global"

  res@trYMaxF            = 3800.0
  res@trYMinF            = 2500.0
  res@lgItemOrder             := (/5,4,3,2,1,0/)
     res@xyDashPatterns       := (/0,0,12,12,12,12/)                 ; dashed line for 2nd)
     res@xyLineThicknesses    := (/3.0,3.0,2.5,2.5,2.5,2.5/)
     res@xyLineColors         := (/"black","slategrey","blue","red","forestgreen","tan4"/)                    ; line color
res@xyExplicitLegendLabels := (/" Obs"," Combined"," VPD"," Temperature", " Forest"," Cropland"/)
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@tiXAxisString          = ""
res@tmXBLabelsOn = False 
  res@tmXBValues             = ispan(2003,2021,3)
  res@tmXBLabels             := ispan(2003,2021,3)
  res@pmLegendDisplayMode    = "Always"
plot(0,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(0,:,:),res)
plot(0,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(0,:,:),res)
  res@pmLegendDisplayMode    = "Never"
  res@gsnLeftString   = plotLabel(1)
  res@trYMaxF            = 100.0
  res@trYMinF            = 30.0
res@gsnCenterString = "Europe"

res@tiYAxisString          =  "" 
plot(1,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(1,:,:),res)
plot(1,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(1,:,:),res)
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@gsnLeftString   = plotLabel(2)

  res@trYMaxF            = 500.0
  res@trYMinF            = 0.0
  res@gsnCenterString = "NAsia"
plot(2,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(2,:,:),res) 
plot(2,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(2,:,:),res)
res@tiYAxisString          =  "" 
res@gsnLeftString   = plotLabel(3)
  res@trYMaxF            = 700.0
  res@trYMinF            = 300.0
res@gsnCenterString = "ESAsia"
plot(3,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(3,:,:),res)
plot(3,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(3,:,:),res)

i=4
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
  res@trYMaxF            = 800.0
  res@trYMinF            = 100.0
res@gsnCenterString = "South America"
res@gsnLeftString   = plotLabel(i)

plot(i,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(i,:,:),res)
plot(i,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(i,:,:),res)

i=5
res@tiYAxisString          =  "" 
  res@trYMaxF            = 300.0
  res@trYMinF            = 100.0
res@gsnCenterString = "North America"
res@gsnLeftString   = plotLabel(i)

plot(i,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(i,:,:),res)
plot(i,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(i,:,:),res)

i=6
res@tmXBLabelsOn = True
res@tiYAxisString          =  "Carbon Emissions (Tg)" 
res@tiXAxisString          = "Year"

  res@trYMaxF            = 2000.0
  res@trYMinF            = 1500.0
res@gsnCenterString = "Africa"
res@gsnLeftString   = plotLabel(i)
plot(i,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(i,:,:),res)
plot(i,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(i,:,:),res)

i=7
res@tiYAxisString          =  "" 
  res@trYMaxF            = 600.0
  res@trYMinF            = 0.0
res@gsnCenterString = "Aus+EQAsia"
res@gsnLeftString   = plotLabel(i)
plot(i,0) = gsn_csm_xy(wks,ispan(2003,2021,1),data(i,:,:),res)
plot(i,1) = gsn_csm_xy(wks,ispan(2003,2021,1),yReg(i,:,:),res)
; do i = 0,7
;   overlay(plot(i,1), plot(i,0))
; end do 
resP@gsnPanelBottom = 0.1
resP@gsnPanelXWhiteSpacePercent = 0.0
resP@gsnPanelYWhiteSpacePercent = 0.0
  gsn_panel(wks, plot(:,0), (/4,2/), resP)




end 