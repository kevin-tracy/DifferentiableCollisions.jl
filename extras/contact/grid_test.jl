

let

    vis = mc.Visualizer()
    mc.open(vis)

    gr = range(-6,6,length = 5)
    # @show length(gr)^2
    grid_xy = vec([SA[i,j] for i = gr, j = gr])


    for i = 1:length(grid_xy)
        sph_p1 = mc.HyperSphere(mc.Point(grid_xy[i]...,0.0), 0.1)
        mc.setobject!(vis["s"*string(i)], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    end

end
