import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg
import scipy.linalg as scylinalg        # non-sparse linear algebra

############################################################
############################################################

def create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogenous Neumann"):
	# defaults to "Homogeneous Dirichlet"

     possible_boundary_conditions = ["Homogeneous Dirichlet","Homogeneous Neumann","Periodic"]

     if not(boundary_conditions in possible_boundary_conditions):
         sys.exit("Boundary conditions need to be either: " +
                  repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
     numbered_pressureCells = -np.ones(Xc.shape,dtype='int64')
     jj_C,ii_C = np.where(pressureCells_Mask==True)
     Np = len(jj_C)

     numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
	 # print(numbered_pressureCells)
	 # quit()
     inv_DxC = 1./Dxc[jj_C,ii_C]
     inv_DyC = 1./Dyc[jj_C,ii_C]

     inv_DxE = 1./(Xc[jj_C,ii_C+1]-Xc[jj_C,ii_C])

     inv_DyN = 1./(Yc[jj_C+1,ii_C]-Yc[jj_C,ii_C])

     inv_DxW = 1./(Xc[jj_C,ii_C]-Xc[jj_C,ii_C-1])
     inv_DyS = 1./(Yc[jj_C,ii_C]-Yc[jj_C-1,ii_C])

     DivGrad = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 # print(numbered_PressureCells)
     iC = numbered_pressureCells[jj_C,ii_C]
     iE = numbered_pressureCells[jj_C,ii_C+1]
     iW = numbered_pressureCells[jj_C,ii_C-1]
     iS = numbered_pressureCells[jj_C-1,ii_C]
     iN = numbered_pressureCells[jj_C+1,ii_C]

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     ## if east node is inside domain
     east_node_mask = (iE!=-1)
     ii_center = iC[east_node_mask]
     ii_east   = iE[east_node_mask]
     inv_dxc_central = inv_DxC[ii_center]
     inv_dxc_east    = inv_DxE[ii_center]
     DivGrad[ii_center,ii_east]   += inv_dxc_central*inv_dxc_east
     DivGrad[ii_center,ii_center] -= inv_dxc_central*inv_dxc_east

     ## if west node is inside domain
     west_node_mask = (iW!=-1)
     ii_center  = iC[west_node_mask]
     ii_west    = iW[west_node_mask]
     inv_dxc_central = inv_DxC[ii_center]
     inv_dxc_west    = inv_DxW[ii_center]
     DivGrad[ii_center,ii_west]   += inv_dxc_central*inv_dxc_west
     DivGrad[ii_center,ii_center] -= inv_dxc_central*inv_dxc_west

	 ## if north node is inside domain
     north_node_mask = (iN!=-1)
     ii_center  = iC[north_node_mask]
     ii_north   = iN[north_node_mask]
     inv_dyc_central  = inv_DyC[ii_center]
     inv_dyc_north    = inv_DyN[ii_center]
     DivGrad[ii_center,ii_north]   += inv_dyc_central*inv_dyc_north
     DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_north

      ## if south node is inside domain
     south_node_mask = (iS!=-1)
     ii_center  = iC[south_node_mask]
     ii_south   = iS[south_node_mask]
     inv_dyc_central  = inv_DyC[ii_center]
     inv_dyc_south    = inv_DyS[ii_center]
     DivGrad[ii_center,ii_south]   += inv_dyc_central*inv_dyc_south
     DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_south

     if boundary_conditions == "Periodic":
         # eb = np.where(east_node_mask==False)[0]
         # wb = np.where(west_node_mask==False)[0]
        eb = np.where(east_node_mask==False)[0]
        wb = np.where(west_node_mask==False)[0]
        sb = np.where(south_node_mask==False)[0]
        nb = np.where(north_node_mask==False)[0]
		# for every east node that is 'just' outside domain
        east_node_mask = (iE==-1)&(iC!=-1)
        ii_center = iC[east_node_mask]
        inv_dxc_central = inv_DxC[ii_center]
        inv_dxc_east    = inv_DxE[ii_center]
        DivGrad[ii_center,ii_center]  -= 1.*inv_dxc_central*inv_dxc_east
        DivGrad[wb,eb] += 1.*inv_dxc_central*inv_dxc_east

		# for every west node that is 'just' outside domain
        west_node_mask = (iW==-1)&(iC!=-1)
        ii_center = iC[west_node_mask]
        inv_dxc_central = inv_DxC[ii_center]
        inv_dxc_west    = inv_DxW[ii_center]
        DivGrad[ii_center,ii_center]  -= 1.*inv_dxc_central*inv_dxc_west
        DivGrad[eb,wb] += 1.*inv_dxc_central*inv_dxc_east

		# for every north node that is 'just' outside domain
        north_node_mask = (iN==-1)&(iC!=-1)
        ii_center = iC[north_node_mask]
        inv_dyc_central  = inv_DyC[ii_center]
        inv_dyc_north    = inv_DyN[ii_center]
        DivGrad[ii_center,ii_center]  -= 1.*inv_dyc_central*inv_dyc_north
        DivGrad[nb,sb] += 1.*inv_dyc_central*inv_dyc_north

		# for every south node that is 'just' outside domain
        south_node_mask = (iS==-1)&(iC!=-1)
        ii_center = iC[south_node_mask]
        inv_dyc_central  = inv_DyC[ii_center]
        inv_dyc_south    = inv_DyS[ii_center]
        DivGrad[ii_center,ii_center]  -= 1.*inv_dyc_central*inv_dyc_south
        DivGrad[sb,nb] += 1.*inv_dyc_central*inv_dyc_south
        # print DivGrad
        # quit()

	 # if Dirichlet boundary conditions are requested, need to modify operator
     if boundary_conditions == "Homogeneous Dirichlet":

		# for every east node that is 'just' outside domain
		east_node_mask = (iE==-1)&(iC!=-1)
		ii_center = iC[east_node_mask]
		inv_dxc_central = inv_DxC[ii_center]
		inv_dxc_east    = inv_DxE[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dxc_central*inv_dxc_east

		# for every west node that is 'just' outside domain
		west_node_mask = (iW==-1)&(iC!=-1)
		ii_center = iC[west_node_mask]
		inv_dxc_central = inv_DxC[ii_center]
		inv_dxc_west    = inv_DxW[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dxc_central*inv_dxc_west

		# for every north node that is 'just' outside domain
		north_node_mask = (iN==-1)&(iC!=-1)
		ii_center = iC[north_node_mask]
		inv_dyc_central  = inv_DyC[ii_center]
		inv_dyc_north    = inv_DyN[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dyc_central*inv_dyc_north

		# for every south node that is 'just' outside domain
		south_node_mask = (iS==-1)&(iC!=-1)
		ii_center = iC[south_node_mask]
		inv_dyc_central  = inv_DyC[ii_center]
		inv_dyc_south    = inv_DyS[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dyc_central*inv_dyc_south

     return DivGrad

# create DDx:
def create_DDx_operator(Dx,Xc,pressureCells_Mask,boundary_conditions="Homogenous Neumann"): # First order upwind difference scheme with second order accurate term

    possible_boundary_conditions = ["Homogeneous Dirichlet","Homogeneous Neumann","Periodic"]

    if not(boundary_conditions in possible_boundary_conditions):
	       sys.exit("Boundary conditions need to be either: " + repr(possible_boundary_conditions))

	# numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
    numbered_pressureCells = -np.ones(Xc.shape,dtype='int64')

    # print np.where(uCells_internal_Mask==True)
    jj_C,ii_C = np.where(pressureCells_Mask==True)
    # jj_C,ii_place = np.where(uCells_internal_Mask==True)
    Np= len(jj_C)
    # Ny = len(ii_C)


    numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening

    # print numbered_pressureCells
	# print(numbered_pressureCells)
	# quit()
	# inv_DxC = 1./Dxc[jj_C,ii_C]
    inv_DxE = 1./(Xc[jj_C,ii_C+1]-Xc[jj_C,ii_C])
    inv_DxW = 1./(Xc[jj_C,ii_C]-Xc[jj_C,ii_C-1])
	# inv_DxC = 1./(Xc[jj_C,ii_C]-Xc[jj_C,ii_C])

    DDx= scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros

    iE = numbered_pressureCells[jj_C,ii_C+1]
    iW = numbered_pressureCells[jj_C,ii_C-1]
    iC = numbered_pressureCells[jj_C,ii_C]
    # print "made it"
	# for First deriv, central difference: (phiE - PhiW)/2/dx
	#
	## if east node is inside domain
    east_node_mask = (iE!=-1)
    ii_center = iC[east_node_mask]
    ii_east   = iE[east_node_mask]
	# inv_dxc_central = inv_DxC[ii_center]
    inv_dxc_east    = inv_DxE[ii_center]
    DDx[ii_center,ii_east]   += .5*inv_dxc_east
    # DDx[ii_center,ii_center] -= inv_dxc_east
     ## if west node is inside domain
    west_node_mask = (iW!=-1)
    ii_center  = iC[west_node_mask]
    ii_west    = iW[west_node_mask]
 	# inv_dxc_central = inv_DxC[ii_center]
    inv_dxc_west    = inv_DxW[ii_center]
    DDx[ii_center,ii_west]   -= .5*inv_dxc_west
    # DDx[ii_center,ii_center] -= inv_dxc_west
    # print DDx
	# if Dirichlet boundary conditions are requested, need to modify operator
    # if boundary_conditions == "Homogeneous Neumann":
    #     east_node_mask = (iE==-1)&(iC!=-1)
    #     west_node_mask = (iW==-1)&(iC!=-1)
    #
    #     ii_center = iC[east_node_mask]
    #     DDx[ii_center,ii_center] -= inv_dxc_east
    #
    #     inv_dxc_east    = inv_DxE[ii_center]

    if boundary_conditions == "Homogeneous Dirichlet":
        eb = np.where(east_node_mask==False)[0]
        wb = np.where(west_node_mask==False)[0]
        # sb = np.where(south_node_mask==False)[0]
        # nb = np.where(north_node_mask==False)[0]
		# for every east node that is 'just' outside domain
		# for every east node that is 'just' outside domain
        east_node_mask = (iE==-1)&(iC!=-1)
        west_node_mask = (iW==-1)&(iC!=-1)

        ii_center = iC[east_node_mask]
        jj_center = iC[west_node_mask]

		# inv_dxc_central = inv_DxC[ii_center]
        inv_dxc_east    = inv_DxE[ii_center]
        inv_dxc_west    = inv_DxW[jj_center]
        DDx[ii_center,ii_center] -= 2.*inv_dxc_west # DDx[ii,ii] that is too far right becomes 1/dx of western most 1/dx
		# for every east node that is 'just' outside domain
        west_node_mask = (iW==-1)&(iC!=-1)

		# inv_dxc_central = inv_DxC[ii_center]

        DDx[jj_center,jj_center] -= 2.*inv_dxc_east # DDx[ii,ii] that is too far left becomes 1/dx of eastern most 1/dx
        # print DDx
		# DDx[ii_center,ii_center]  -= 2.*inv_dxc_central
    # print DDx
    return DDx

def create_DDy_operator(Dyc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Neumann"):

    possible_boundary_conditions = ["Homogeneous Dirichlet","Homogeneous Neumann","Periodic"]

    if not(boundary_conditions in possible_boundary_conditions):
        sys.exit("Boundary conditions need to be either: " + repr(possible_boundary_conditions))

	# numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
    numbered_pressureCells = -np.ones(Yc.shape,dtype='int64')

    # print np.where(uCells_internal_Mask==True)
    jj_C,ii_C = np.where(pressureCells_Mask==True)
    # jj_C,ii_place = np.where(uCells_internal_Mask==True)
    Np = len(jj_C)



    numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
    # print numbered_pressureCells

    inv_DyS = 1./(Yc[jj_C,ii_C]-Yc[jj_C-1,ii_C])
    inv_DyN = 1./(Yc[jj_C+1,ii_C]-Yc[jj_C,ii_C])
	# inv_DxC = 1./(Xc[jj_C,ii_C]-Xc[jj_C,ii_C])

    DDy= scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros

	# iE = numbered_pressureCells[jj_C,ii_C+1]
	# iW = numbered_pressureCells[jj_C,ii_C-1]
    iC = numbered_pressureCells[jj_C,ii_C]
    iN = numbered_pressureCells[jj_C+1,ii_C]
    iS = numbered_pressureCells[jj_C-1,ii_C]
	# for First deriv, central difference: (phiE - PhiW)/2/dx
	#
	## if east node is inside domain
    south_node_mask = (iS!=-1)
    ii_center = iC[south_node_mask]
    ii_south   = iS[south_node_mask]
	# inv_dxc_central = inv_DxC[ii_center]
    inv_dxc_south = inv_DyS[ii_center]
    DDy[ii_center,ii_south] -= .5*inv_dxc_south
    # DDy[ii_center,ii_center] -= inv_dxc_south

     ## if west node is inside domain
    north_node_mask = (iN!=-1)
    ii_center = iC[north_node_mask]
    ii_north = iN[north_node_mask]
 	# inv_dxc_central = inv_DxC[ii_center]
    inv_dxc_north = inv_DyN[ii_center]
    # inv_dyc_central  = inv_DyC[ii_center]
    DDy[ii_center,ii_north]  += .5*inv_dxc_north
    # DDy[ii_center,ii_center] -= inv_dxc_north
    # print "made it"
    # print DDy
	# if Dirichlet boundary conditions are requested, need to modify operator

        # ii_center = iC[north_node_mask]

    if boundary_conditions == "Homogeneous Dirichlet":
		# sb = np.where(south_node_mask==False)[0]
        # nb = np.where(north_node_mask==False)[0]
        # sb = np.where(south_node_mask==False)[0]
        # nb = np.where(north_node_mask==False)[0]
		# for every east node that is 'just' outside domain
		# for every east node that is 'just' outside domain
        south_node_mask = (iS==-1)&(iC!=-1)
        north_node_mask = (iN==-1)&(iC!=-1)

        jj_center = iC[south_node_mask]
        ii_center = iC[north_node_mask]

		# inv_dxc_central = inv_DxC[ii_center]
        inv_dxc_south    = inv_DyS[ii_center]
        inv_dxc_north    = inv_DyN[jj_center]
        DDy[ii_center,ii_center]  -= 2.*inv_dxc_south # DDx[ii,ii] that is too north becomes 1/dx of southern most 1/dx
		# for every east node that is 'just' outside domain

		# inv_dxc_central = inv_DxC[ii_center]

        DDy[jj_center,jj_center] -= 2.*inv_dxc_north # DDx[ii,ii] that is too far south becomes 1/dx of northern most 1/dx

    return DDy
