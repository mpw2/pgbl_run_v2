#include "cti_solver_ig.hpp"
#include "cti_solver_ig_flux.hpp"
#include "cti_solver_register.hpp" // for checkInit
#include "cti_solver_charlesCommon.hpp"
#include "wm/AlgebraicWM.hpp"
#include "wm/AlgebraicKESWM.hpp"
#include "AlgebraicWM_spalding.hpp"
#include "cti_solver_cht.hpp"
#include "cti_solver_ig_gpu.hpp"
#include "cti_solver_ig_bcs_kernels.hpp" 
#include "cti_solver_sgs_gpu.hpp"
#include "cti_solver_common_kernels.hpp"
#include "cti_utils_profileReader.hpp"
#include "DataExchanger.hpp"
#include "BoundaryLayerDataExchanger.hpp"

namespace cti {
  namespace solver {
    namespace ig {

      vector<IdealGasBc*> bcVec;
      cti_ffp * force_work_arr_d = NULL; // work array for device force calcs.. 
      int nfw                    = -1;   // n force work... 
      
      IdealGasBc::IdealGasBc(BfZone* zone_ptr) {
        this->zone_ptr = zone_ptr;
        registerFunc(&IdealGasBc::force,this,zone_ptr->name+":force");
        registerFunc(&IdealGasBc::tau_wall,this,zone_ptr->name+":tau_wall");
      }

      CtiData *IdealGasBc::force(vector<CtiData*>& argVec,const bool b_eval) {
        //COUT("IdealGasBc::force() " << eval_func_name);
        // ------------------------------------------------------------
        // force() returns the force on all boundary faces in a given 
        // zone. It takes one parameter indicating if the total force
        // is desired or a portion of the force, i.e. convective, pressure
        // or viscous.
        // ------------------------------------------------------------
        ForceType f_type = ForceType::TOTAL_FORCE;  //default, compute total force
        if (argVec.size()==1){
          //component of force to compute...convective, pressure or viscous
          if (argVec[0]->getDatatype()!=I_DATA){  //TODO also allow strings matching f_type enum in cti_solver_forces
            eval_cerr << "force(ForceType) expects type argument as I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
            return NULL;
          }
          int f_type_int = argVec[0]->i();
          switch (f_type_int) {
            case (static_cast<int>(ForceType::TOTAL_FORCE)):
            case (static_cast<int>(ForceType::CONVECTIVE_FORCE)):
            case (static_cast<int>(ForceType::PRESSURE_FORCE)):
            case (static_cast<int>(ForceType::VISCOUS_FORCE)):
              f_type = static_cast<ForceType>(f_type_int);
              break;
            default:
              eval_cerr << "argument for force(ForceType) is out of range, got " << f_type_int << endl;
              return NULL;
          }
        }
        else if (argVec.size()>1){
          eval_cerr << "force() accepts up to 1 argument: got " << argVec.size() << endl;
          return NULL;
        }
         
        CtiData *rval = new CtiData(DN3_DATA,zone_ptr->nbf,b_eval);
        if (b_eval) { 
          double (*rval_dn3)[3] = rval->dn3();
          force(rval_dn3,f_type); //each boundary condition must implement
          /*
          if (f_type == ForceType::TOTAL_FORCE || f_type == ForceType::PRESSURE_FORCE){
            //return pressure force in dp form...
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 rval_dn3[ibf][i] -= p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          */
        }
        return rval;
      }

      CtiData *IdealGasBc::tau_wall(vector<CtiData*>& argVec,const bool b_eval) {
        //COUT("IdealGasBc::tau_wall() " << eval_func_name);
        // ------------------------------------------------------------
        // tau_wall() returns the viscous shear stress. 
        // ------------------------------------------------------------

        int comp = -1;
        if (argVec.size() == 1) {
          // component of force to compute...convective, pressure or viscous
          if (argVec[0]->getDatatype() != I_DATA){  
            eval_cerr << "tau_wall() argument expects I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
            return NULL;
          }
          comp = argVec[0]->i();
          if (( comp < 0) || (comp >= 3) ) {
            eval_cerr << "tau_wall() integer argument should be in [0:3), got " << comp << endl;
            return NULL;
          }
        }
        else if (argVec.size() > 1) {
          eval_cerr << "tau_wall() accepts up to 1 argument: got " << argVec.size() << endl;
          return NULL;
        }
         
        CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
        if (b_eval) { 
          double (*dn3)[3] = new double[zone_ptr->nbf][3];
          force(dn3,ForceType::VISCOUS_FORCE); // each boundary condition must implement 
          double *rval_dn = rval->dn();
          // default is magnitude
          if (comp == -1) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              double tau[3]; FOR_I3 tau[i] = dn3[ibf][i]/zone_ptr->area_bf[ibf];
              rval_dn[ibf] = MAG(tau);
            }
          }
          else {
            assert((comp >= 0)&&(comp < 3));
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              rval_dn[ibf] = dn3[ibf][comp]/zone_ptr->area_bf[ibf];
            }
          }
          //cti::utils::dumpRange(rval_dn,zone_ptr->nbf,"rval_dn");
          delete[] dn3;
        }
        return rval;
      }

      //wall force helper templates
      
      template <class T>
      inline void force_bf_tauvar(double (*rval_dn3)[3], const ForceType f_type, T* bc) {
      
        for (int ibf = 0; ibf < bc->zone_ptr->nbf; ++ibf) 
          FOR_I3 rval_dn3[ibf][i] = 0.0;
 
        if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ){
          for (int ibf = 0; ibf < bc->zone_ptr->nbf; ++ibf){
            const int icv0 = bc->zone_ptr->cvobf[ibf];
            FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*bc->zone_ptr->n_bf[ibf][i];// dp
          }
        }
        if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ){
          // tau_wall has already been computed at this point, but we'll assume
          // that the wall stress is instantaneously aligned with the velocity vector
          for (int ibf = 0; ibf < bc->zone_ptr->nbf; ++ibf){
            const int icv0 = bc->zone_ptr->cvobf[ibf];
            const double du[3] = DIFF(u[icv0],bc->u_bc[ibf]);
            double u_mag = DOT_PRODUCT(du,du);
            if (u_mag > 0.0){
              u_mag = sqrt(max(0.0,u_mag));
              FOR_I3 rval_dn3[ibf][i] += bc->tau_wall[ibf]*du[i]/u_mag*bc->zone_ptr->area_bf[ibf]; 
            }
          }
        }
      }

      void ensureForceWorkArrayDevice(const int n) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

        if ( (force_work_arr_d == NULL) || (nfw < n) ) { 

          if ( force_work_arr_d) { 
          
            cti::utils::freeOnDevice(force_work_arr_d);
            force_work_arr_d = NULL;
          
          } 

          cti::utils::initOnDevice(force_work_arr_d, n);
          nfw = n;

        }
#endif

      } 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

      template<typename T>
      __global__ 
      void addPressureForceDevice( T *__restrict__ force_arr, 
                                   T *__restrict__ p, 
                                   int *__restrict__ cvobf, 
                                   T *__restrict__ n_bf,
                                   const T p_ref, 
                                   const int nbf) { 

        const int ibf = blockIdx.x * blockDim.x + threadIdx.x;

        if ( ibf < nbf) {

          const int icv = cvobf[ibf];
          for (int i = 0; i < 3; ++i) 
            force_arr[3*ibf+i] += (p[icv] - p_ref) * n_bf[3*ibf+i]; // dp form... 

        } 
      
      } 

      template<typename T>
      __global__ 
      void addViscousForceTauVarDevice( T *__restrict__ force_arr, 
                                        T *__restrict__ u, 
                                        T *__restrict__ tau_wall, 
                                        T *__restrict__ u_bc, 
                                        int *__restrict__ cvobf, 
                                        T *__restrict__ area_bf,
                                        const int nbf) { 

        const int ibf = blockIdx.x * blockDim.x + threadIdx.x;

        if ( ibf < nbf) {

          const int icv = cvobf[ibf];
          T du[3];
          T u_mag = 0.0;

          for (int i = 0; i < 3; ++i)  { 
            du[i]  = u[3*icv+i] - u_bc[3*ibf+i];
            u_mag += du[i]*du[i];
          }

          u_mag = sqrtf(u_mag);
          
          if ( u_mag > 0.0) { 
            
            for (int i = 0; i < 3; ++i) 
              force_arr[3*ibf+i] += tau_wall[ibf] * du[i]/u_mag * area_bf[ibf];

          } 
        } 

      } 
      
      template<typename T>
      __global__ 
      void addViscousForceOscTauVarDevice( T *__restrict__ force_arr, 
                                        T *__restrict__ u, 
                                        T *__restrict__ tau_wall, 
                                        T *__restrict__ u_bc, 
                                        const T t,
                                        const T osc_period,
                                        int *__restrict__ cvobf, 
                                        T *__restrict__ area_bf,
                                        const int nbf) { 

        const int ibf = blockIdx.x * blockDim.x + threadIdx.x;

        if ( ibf < nbf) {

          const int icv = cvobf[ibf];
          T du[3];
          T u_mag = 0.0;
          
          T osc_fac = 1.0f;
          if (osc_period > 0.0f) {
            osc_fac = sin(t/osc_period);
          }

          for (int i = 0; i < 3; ++i)  { 
            du[i]  = u[3*icv+i] - osc_fac*u_bc[3*ibf+i];
            u_mag += du[i]*du[i];
          }

          u_mag = sqrtf(u_mag);
          
          if ( u_mag > 0.0) { 
            
            for (int i = 0; i < 3; ++i) 
              force_arr[3*ibf+i] += tau_wall[ibf] * du[i]/u_mag * area_bf[ibf];

          } 
        } 

      } 


#endif

      template <class T>
      inline void force_bf_tauvar_device(double (*rval_dn3)[3], const ForceType f_type,T* bc) {
      
#if defined(BUILD_CUDA) || defined(BUILD_HIP)

        if ( bc->zone_ptr->nbf > 0 ) { 

          // we need a temporary storage container on the device that is a 
          // vector nbf record... 

          BfZone * zone_ptr = bc->zone_ptr;

          ensureForceWorkArrayDevice(3*zone_ptr->nbf);
          assert( force_work_arr_d);
          cti::utils::deviceMemset(force_work_arr_d, 0, 3*zone_ptr->nbf*sizeof(cti_ffp));

          if ( (f_type == ForceType::TOTAL_FORCE   ) || 
               ( f_type == ForceType::PRESSURE_FORCE)  ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size; 

            LAUNCH_GPU_KERNEL( addPressureForceDevice, grid_size, block_size, 
                               force_work_arr_d, p_f_d, 
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d, 
                               (cti_ffp) p_ref, zone_ptr->nbf);

          }

          if ( (f_type == ForceType::TOTAL_FORCE   ) || 
               (f_type == ForceType::VISCOUS_FORCE))     { 


            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size; 

            LAUNCH_GPU_KERNEL( addViscousForceTauVarDevice, grid_size, block_size, 
                               force_work_arr_d, u_f_d, bc->tau_wall_d, bc->u_bc_d,
                               zone_ptr->cvobf_d, zone_ptr->area_bf_d, 
                               zone_ptr->nbf);


          } 

          cti::utils::copyToHost((double*)rval_dn3, force_work_arr_d, 3*zone_ptr->nbf);

        } 

#endif

      }


      /*
      template <class T>
      inline void force_bf_taufunc(double (*rval_dn3)[3], const ForceType f_type,T* bc) {
      
        for (int ibf = 0; ibf < bc->zone_ptr->nbf; ++ibf) 
          FOR_I3 rval_dn3[ibf][i] = 0.0;
 
        if ( (f_type==ForceType::TOTAL_FORCE) || 
             (f_type==ForceType::PRESSURE_FORCE) ){
          for (int ibf = 0; ibf < bc->zone_ptr->nbf; ++ibf){
            const int icv0 = bc->zone_ptr->cvobf[ibf];
            FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*bc->zone_ptr->n_bf[ibf][i];
          }
        }

        if ( (f_type==ForceType::TOTAL_FORCE) ||
             (f_type==ForceType::VISCOUS_FORCE) ){
          // tau_wall has already been computed at this point, but we'll assume
          // that the wall stress is instantaneously aligned with the velocity vector
          CtiData * tau_wall = eval(bc->zone_ptr->name+":tau_wall()");
          if (tau_wall && tau_wall->getDatatype()==DN_DATA){
            double * tau_wall_dn = tau_wall->dn();
            for (int ibf = 0; ibf < bc->zone_ptr->nbf; ++ibf){
              const int icv0 = bc->zone_ptr->cvobf[ibf];
              const double du[3] = DIFF(u[icv0],bc->u_bc[ibf]);
              double u_mag = DOT_PRODUCT(du,du);
              u_mag = sqrt(max(0.0,u_mag));
              if (u_mag > 0.0){
                FOR_I3 rval_dn3[ibf][i] += tau_wall_dn[ibf]*du[i]/u_mag*bc->zone_ptr->area_bf[ibf]; 
                FOR_I3 rval_dn3[ibf][i] += 0.0;
              }
            }
          }
        }
      }
      */


      class SlipBc : public IdealGasBc {
      public:
	SlipBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  if (mpi_rank == 0) cout << "SlipBc: " << param->str() << endl;
	  zone_ptr->lb_cost = 1;
	}
	~SlipBc() {}
	void initData() {}
	void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    FOR_I3 rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	  }
	}
	void rkStep(const double *rk_wgt,const int rk_stage) {}
	void preLoadBalance() {}
	void postLoadBalance() {}
        void query() {}
      
        void addBoundaryFluxGpu(cti_ffp * rhs, const int rk_stage, 
                                   gpuStream_t& stream) { 

#if defined(BUILD_CUDA)|| defined(BUILD_HIP)
          
          if ( zone_ptr->nbf > 0 ) { 

            // recall that the boundary condition needs to encapsulate 
            // fluxes in delta form... 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addSlipIgBcFlux, grid_size, 
                                      block_size, stream, rhs, sp_vol_f_d,
                                      u_f_d, p_f_d, h_f_d, zone_ptr->cvobf_d,
                                      zone_ptr->n_bf_d, zone_ptr->nbf);  

          } 

#endif
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {}

        void force(double (*rval_dn3)[3], const ForceType f_type) {
      

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 


          if ( (f_type == ForceType::PRESSURE_FORCE) || 
               (f_type == ForceType::TOTAL_FORCE) ) {

            if ( zone_ptr->nbf > 0 ) { 

              ensureForceWorkArrayDevice(3*zone_ptr->nbf);
              assert( force_work_arr_d);
              cti::utils::deviceMemset(force_work_arr_d,0,3*zone_ptr->nbf*sizeof(cti_ffp));

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size;

              LAUNCH_GPU_KERNEL( addPressureForceDevice, grid_size, block_size, 
                                 force_work_arr_d, p_f_d, 
                                 zone_ptr->cvobf_d, zone_ptr->n_bf_d, 
                                 (cti_ffp) p_ref, zone_ptr->nbf);

        
              cti::utils::copyToHost((double*)rval_dn3, force_work_arr_d, 3*zone_ptr->nbf);
            }
          
          } else { 

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
              for (int i = 0; i < 3; ++i) 
                rval_dn3[ibf][i] = 0.0;

          } 

#else

          if ( (f_type == ForceType::PRESSURE_FORCE)  || 
               (f_type == ForceType::TOTAL_FORCE)         ) {

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] = (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
          else{
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
              FOR_I3 rval_dn3[ibf][i] = 0.0;
          }

#endif
        }
                                                                     
      };

      class WallIsothermalBc : public IdealGasBc {
      private:

	double factor;

	double * u_wall;
	double * xrot_wall;
	double * axisrot_wall;
	double omega_wall;

	bool b_stationary;

      public:

	double (*u_bc)[3];
	double *tau_wall;
        double *q_wall;

        cti_ffp * u_bc_d;
        cti_ffp * tau_wall_d;
        cti_ffp * q_wall_d;

        double T_wall;

	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WALL_ISOTHERMAL T_WALL <double> [FACTOR <double>]"
	    "\n  [TRANSLATE <double> <double> <double>]"
	    "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
	    "\n  [STATIONARY]";
	}

	WallIsothermalBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  
          if (mpi_rank == 0) cout << "WallIsothermalBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  zone_ptr->lb_cost = 50;
	  
          u_bc       = NULL;
	  tau_wall   = NULL;
          q_wall     = NULL;
          u_bc_d     = NULL;
          tau_wall_d = NULL;
          q_wall_d   = NULL;

	  u_wall       = NULL;
	  xrot_wall    = NULL;
	  axisrot_wall = NULL;
	  omega_wall   = BIG_DOUBLE;

	  b_stationary = false;

	  registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall", DEP_DATA); // no need to read/write a laminar wall
	  registerData(q_wall,zone_ptr->nbf,zone_ptr->name+":q_wall", DEP_DATA); // no need to read/write a laminar wall
	  registerFunc(&WallIsothermalBc::y_plus,this,zone_ptr->name+":y_plus");

	  factor = 0.0;
          T_wall = -1.0; // invalid as default.

	  int iarg = 1;
	  while ( iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ( token == "FACTOR") {
	      factor = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > FACTOR: " << factor << endl;
	    }
            else if ( token == "T_WALL") { 
              T_wall = param->getDouble(iarg++);
              if ( mpi_rank == 0 ) cout << " > T_WALL : " << T_wall << endl;
            }
	    else if ( token == "TRANSLATE") {
	      assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	    }
            else if ( token == "ROTATING") {
	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            } 
            else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString()); 
	    }
	  }

          if ( T_wall < 0.0) { 
            CERR(" > must specify a valid T_WALL for zone: " << zone_ptr->getName());
          } 

	}

	~WallIsothermalBc() {

	  DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
          DELETE(q_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);
	
        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {
	
          // init tau_wall...
	  
          assert(tau_wall);
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    tau_wall[ibf] = 0.0;
	  }
	  
          // init q_wall
          
          assert(q_wall);
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 
            q_wall[ibf] = 0.0;
          } 

          // init u_bc...
	  
          assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];	  
	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)){
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) {
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert (u_bc_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          
          assert (tau_wall_d == NULL);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);

          assert( q_wall_d == NULL);
          cti::utils::initAndCopyToDevice(q_wall_d,q_wall,zone_ptr->nbf);

#endif

	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
          const double cp = R_gas*gamma/(gamma-1.0);

          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    tau_wall[ibf] = 0.0;
	    if (zone_ptr->area_bf[ibf] > 0.0) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double visc_coeff = (mu_lam[icv]+factor*mu_sgs[icv])*zone_ptr->area_over_delta_bf[ibf];
	      double visc_work = 0.0;
	      FOR_I3 {
		const double tauijnj = visc_coeff*(u[icv][i] - u_bc[ibf][i]);
		tau_wall[ibf] += tauijnj*tauijnj;
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i] + tauijnj;
		visc_work += u_bc[ibf][i]*tauijnj;
	      }

	      // TODO: need area and delta separately to avoid all this checking...
	      tau_wall[ibf] = sqrt(tau_wall[ibf])/zone_ptr->area_bf[ibf];
              rhs[icv][4]      -= visc_work;

              double delta_bf   = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
              q_wall[ibf]       = loc_lam[icv]*cp*(T[icv]-T_wall)/delta_bf;
              rhs[icv][4]      -= q_wall[ibf]*zone_ptr->area_bf[ibf];

	    }
	  }
	}
	
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            const cti_ffp h_wall = R_gas*gamma/(gamma-1.0)*T_wall;

            LAUNCH_GPU_KERNEL_STREAM( addWallIsothermalFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, loc_lam_d, u_bc_d, tau_wall_d,
                                      q_wall_d, h_wall,
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      zone_ptr->nbf);

          }

#endif

        }

 

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {
          
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d);     u_bc_d     = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;
          cti::utils::freeOnDevice(q_wall_d);   q_wall_d   = NULL;
#endif
        }
	
        void postLoadBalance() {
          initData();
        }
      
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        
        void syncToHost() {
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
          cti::utils::copyToHost(q_wall,q_wall_d,zone_ptr->nbf);
#endif	
	}

        void query() {

          double my_buf[4] = {0.0,0.0,0.0,0.0};

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];
            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1       = zone_ptr->area_bf[ibf] / 
                                      zone_ptr->area_over_delta_bf[ibf];
              const double y_plus   = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]            += zone_ptr->area_bf[ibf];
              my_buf[1]            += zone_ptr->area_bf[ibf] * tau_wall[ibf];
              my_buf[2]            += zone_ptr->area_bf[ibf] * y_plus;
              my_buf[3]            += zone_ptr->area_bf[ibf] * q_wall[ibf];

            } 
          } 

          double buf[4];
          MPI_Reduce(my_buf,buf,4,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:avg(y_plus) " << buf[2]/buf[0] 
                 << " 12:int(q_wall)dA " << buf[3]
		 << " 14:area " << buf[0]
                 << endl; 
          }
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          force_bf_tauvar_device(rval_dn3,f_type,this);
#else
          force_bf_tauvar(rval_dn3,f_type,this); 
#endif
        }

      
      };
 
      class WallHeatFluxBc : public IdealGasBc {
      public:  


        double qwall_nom;
	double * u_wall;
	double * xrot_wall;
	double * axisrot_wall;
	double omega_wall;

	bool b_stationary;

	double (*u_bc)[3];
	double *tau_wall;

        cti_ffp * u_bc_d;
        cti_ffp * tau_wall_d;


	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WALL_HEAT_FLUX Q_WALL <double>"
	    "\n  [TRANSLATE <double> <double> <double>]"
	    "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
	    "\n  [STATIONARY]";
	}

	WallHeatFluxBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  
          if (mpi_rank == 0) cout << "WallHeatFluxBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  zone_ptr->lb_cost = 50;
	  
          u_bc       = NULL;
	  tau_wall   = NULL;
          u_bc_d     = NULL;
          tau_wall_d = NULL;

	  u_wall       = NULL;
	  xrot_wall    = NULL;
	  axisrot_wall = NULL;
	  omega_wall   = BIG_DOUBLE;

	  b_stationary = false;

	  registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall", DEP_DATA); // no need to read/write a laminar wall
	  registerFunc(&WallHeatFluxBc::y_plus,this,zone_ptr->name+":y_plus");

          qwall_nom = BIG_DOUBLE; 

	  int iarg = 1;
	  while ( iarg < param->size()) {
	    string token = param->getUCString(iarg++);
            if ( token == "Q_WALL") { 
              qwall_nom = param->getDouble(iarg++);
              if ( mpi_rank == 0 ) cout << " > Q_WALL : " << qwall_nom << endl;
            }
	    else if ( token == "TRANSLATE") {
	      assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	    }
            else if ( token == "ROTATING") {
	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            } 
            else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString()); 
	    }
	  }

          if ( qwall_nom == BIG_DOUBLE ) { 
            CERR(" > must specify a valid Q_WALL for zone: " << zone_ptr->getName());
          } 

	}

	~WallHeatFluxBc() {

	  DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);
	
        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {
	
          // init tau_wall...
	  
          assert(tau_wall);
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    tau_wall[ibf] = 0.0;
	  }
	  
          // init u_bc...
	  
          assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];	  
	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)){
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) {
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert (u_bc_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          
          assert (tau_wall_d == NULL);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);

#endif

	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    tau_wall[ibf] = 0.0;
	    if (zone_ptr->area_bf[ibf] > 0.0) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double visc_coeff = mu_lam[icv]*zone_ptr->area_over_delta_bf[ibf];
	      double visc_work = 0.0;
	      FOR_I3 {
		const double tauijnj = visc_coeff*(u[icv][i] - u_bc[ibf][i]);
		tau_wall[ibf] += tauijnj*tauijnj;
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i] + tauijnj;
		visc_work += u_bc[ibf][i]*tauijnj;
	      }

	      // TODO: need area and delta separately to avoid all this checking...
	      tau_wall[ibf] = sqrt(tau_wall[ibf])/zone_ptr->area_bf[ibf];
              rhs[icv][4]      -= visc_work;

              rhs[icv][4]      -= qwall_nom*zone_ptr->area_bf[ibf];

	    }
	  }
	}
	
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addWallHeatFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, loc_lam_d, u_bc_d, tau_wall_d,
                                      (cti_ffp) qwall_nom, 
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      zone_ptr->nbf);

          }

#endif

        }

 

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {
          
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d);     u_bc_d     = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;
#endif
        }
	
        void postLoadBalance() {
          initData();
        }
      
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        
        void syncToHost() {
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
#endif	
	}

        void query() {

          double my_buf[4] = {0.0,0.0,0.0,0.0};

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];
            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1       = zone_ptr->area_bf[ibf] / 
                                      zone_ptr->area_over_delta_bf[ibf];
              const double y_plus   = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]            += zone_ptr->area_bf[ibf];
              my_buf[1]            += zone_ptr->area_bf[ibf] * tau_wall[ibf];
              my_buf[2]            += zone_ptr->area_bf[ibf] * y_plus;
              my_buf[3]            += zone_ptr->area_bf[ibf] * qwall_nom; 

            } 
          } 

          double buf[4];
          MPI_Reduce(my_buf,buf,4,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:avg(y_plus) " << buf[2]/buf[0] 
                 << " 12:int(q_wall)dA " << buf[3] 
		 << " 14:area " << buf[0]
		 << endl; 
          }
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          force_bf_tauvar_device(rval_dn3,f_type,this);
#else
          force_bf_tauvar(rval_dn3,f_type,this); 
#endif
        }

      
      };
 

      class WallAdiabaticBc : public IdealGasBc {
      private:

	double factor;

	double * u_wall;
	double * xrot_wall;
	double * axisrot_wall;
	double omega_wall;

	bool b_stationary;

        BoundaryLayerDataExchanger * blde;
        int nbl;
        double l_bl;
        bool b_blde_params;
        const double bl_thresh = 0.99;
        
        double * bl_delta;
        double * bl_delta_star;
        double * bl_theta;
        double * bl_pt_error;
        double * bl_u_edge;

      public:

	double (*u_bc)[3];
	double *tau_wall;

        cti_ffp * u_bc_d;
        cti_ffp * tau_wall_d;

	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WALL_ADIABATIC [FACTOR <double>]"
	    "\n  [TRANSLATE <double> <double> <double>]"
	    "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
	    "\n  [STATIONARY]";
	}

	WallAdiabaticBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  
          if (mpi_rank == 0) cout << "WallAdiabaticBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  zone_ptr->lb_cost = 50;
	  
          u_bc       = NULL;
	  tau_wall   = NULL;
          u_bc_d     = NULL;
          tau_wall_d = NULL;

	  u_wall       = NULL;
	  xrot_wall    = NULL;
	  axisrot_wall = NULL;
	  omega_wall   = BIG_DOUBLE;

	  b_stationary = false;
          
          blde = NULL;
          nbl = BIG_INT;
          l_bl = BIG_DOUBLE;
          b_blde_params = false;

          bl_delta = NULL;
          bl_delta_star = NULL;
          bl_theta = NULL;
          bl_pt_error = NULL;
          bl_u_edge = NULL;

	  registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall", DEP_DATA); // no need to read/write a laminar wall
	  registerFunc(&WallAdiabaticBc::y_plus,this,zone_ptr->name+":y_plus");

	  factor = 0.0;

	  int iarg = 1;
	  while ( iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ( token == "FACTOR") {
	      factor = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > FACTOR: " << factor << endl;
	    }
	    else if ( token == "TRANSLATE") {
	      assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	    }
            else if ( token == "ROTATING") {
	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            } 
            else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString()); 
	    }
	  }

          // Parse parameters for the boundary layer data exchanger

          Param * blde_param = getParam("BLDE");
          if ( blde_param) {
            iarg = 0;
            while ( iarg < blde_param->size()) {
              string token = blde_param->getString(iarg++);
              if (token == "NBL") {
                nbl = blde_param->getInt(iarg++);
              }
              else if ( token == "L_BL") {
                l_bl = blde_param->getDouble(iarg++);
              }
              else {
                CERR("unrecognized token " << token <<
                     " during parsing of BLDE in bc:\n" << param->str() << helpString());
              }
            }
          }
          if ( l_bl != BIG_DOUBLE && nbl != BIG_INT) {
            b_blde_params = true;

            registerData(bl_delta,zone_ptr->nbf,zone_ptr->name+":bl_delta",DEP_DATA);
            registerData(bl_delta_star,zone_ptr->nbf,zone_ptr->name+":bl_delta_star",DEP_DATA);
            registerData(bl_theta,zone_ptr->nbf,zone_ptr->name+":bl_theta",DEP_DATA);
            registerData(bl_pt_error,zone_ptr->nbf,zone_ptr->name+":bl_pt_error",DEP_DATA);
            registerData(bl_u_edge,zone_ptr->nbf,zone_ptr->name+":bl_u_e",DEP_DATA);
          }

	}
	~WallAdiabaticBc() {
	  DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);

          if (blde) 
            delete blde;

	}
	CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {
          
          // init boundary layer data exchanger
          
          if (b_blde_params) {
            assert(blde == NULL);
            blde = new BoundaryLayerDataExchanger(zone_ptr,nbl,l_bl);

            assert(bl_delta);
            assert(bl_delta_star);
            assert(bl_theta);
            assert(bl_pt_error);
            assert(bl_u_edge);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              bl_delta[ibf] = 0.0;
              bl_delta_star[ibf] = 0.0;
              bl_theta[ibf] = 0.0;
              bl_pt_error[ibf] = 0.0;
              bl_u_edge[ibf] = 0.0;
            }
          }
	
          // init tau_wall...
	  
          assert(tau_wall);
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    tau_wall[ibf] = 0.0;
	  }
	  
          // init u_bc...
	  
          assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];	  
	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)){
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) {
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert (u_bc_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          assert (tau_wall_d == NULL);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);


#endif

	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    tau_wall[ibf] = 0.0;
	    if (zone_ptr->area_bf[ibf] > 0.0) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double visc_coeff = (mu_lam[icv]+factor*mu_sgs[icv])*zone_ptr->area_over_delta_bf[ibf];
	      double visc_work = 0.0;
	      FOR_I3 {
		const double tauijnj = visc_coeff*(u[icv][i] - u_bc[ibf][i]);
		tau_wall[ibf] += tauijnj*tauijnj;
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i] + tauijnj;
		visc_work += u_bc[ibf][i]*tauijnj;
	      }
	      // TODO: need area and delta separately to avoid all this checking...
	      tau_wall[ibf] = sqrt(tau_wall[ibf])/zone_ptr->area_bf[ibf];
	      rhs[icv][4]      -= visc_work;
	    }
	  }
	}
	
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addWallAdiabaticFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, u_bc_d, tau_wall_d,
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      zone_ptr->nbf);

          }

#endif

        }

 

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {
          
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;
#endif
        }
	
        void postLoadBalance() {
          initData();
        }
      
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
#endif	
          
          if (b_blde_params) {
            blde->computeBlFromPt(bl_delta,bl_delta_star,bl_theta,bl_u_edge,bl_thresh);
            blde->computeBlPtError(bl_pt_error);
          }

	}

        void query() {

          double my_buf[3] = {0.0,0.0,0.0};
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];

            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1  = zone_ptr->area_bf[ibf] / 
                                 zone_ptr->area_over_delta_bf[ibf];

              const double y_plus = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]          += zone_ptr->area_bf[ibf];
              my_buf[1]          += zone_ptr->area_bf[ibf]*tau_wall[ibf];
              my_buf[2]          += zone_ptr->area_bf[ibf]*y_plus;


            } 
          }

          double buf[3];
          MPI_Reduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:avg(y_plus) " << buf[2]/buf[0] 
                 << endl;
          }
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          force_bf_tauvar_device(rval_dn3,f_type,this);
#else 
          force_bf_tauvar(rval_dn3,f_type,this); 
#endif
        }
      };
      

      class WmAlgAdiabaticBc : public IdealGasBc {
      private:

	double factor;

	double * u_wall;
	double * xrot_wall;
	double * axisrot_wall;
	double omega_wall;
        double osc_period;

	bool b_stationary;
        bool b_use_bfocv;

        BoundaryLayerDataExchanger * blde;
        int nbl;
        double l_bl;
        bool b_blde_params;
        const double bl_thresh = 0.99;

        double * bl_delta;
        double * bl_delta_star;
        double * bl_theta;
        double * bl_pt_error;
        double * bl_u_edge;

      public:

	double (*u_bc)[3];
	double *tau_wall;

        cti_ffp * u_bc_d;
	cti_ffp * tau_wall_d;

	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WM_ALG_ADIABATIC [FACTOR <double>]"
	    "\n  [TRANSLATE <double> <double> <double>]"
	    "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
	    "\n  [STATIONARY]";
	}

	WmAlgAdiabaticBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {

	  if (mpi_rank == 0) cout << "WmAlgAdiabaticBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  zone_ptr->lb_cost = 200;
	  
          u_bc       = NULL;
	  tau_wall   = NULL; 
          u_bc_d     = NULL;
	  tau_wall_d = NULL;
	  
	  u_wall       = NULL;
	  xrot_wall    = NULL;
	  axisrot_wall = NULL;
	  omega_wall   = BIG_DOUBLE;
          osc_period   = -1.0;

	  b_stationary = false;

          blde = NULL;
          nbl = BIG_INT;
          l_bl = BIG_DOUBLE;
          b_blde_params = false;

          bl_delta = NULL;
          bl_delta_star = NULL;
          bl_theta = NULL;
          bl_pt_error = NULL;
          bl_u_edge = NULL;

          registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall", READ_DATA | WRITE_DATA);
	  registerFunc(&WmAlgAdiabaticBc::y_plus,this,zone_ptr->name+":y_plus");

	  factor = 1.0;

	  int iarg = 1;
	  while ( iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ( token == "FACTOR") {
	      factor = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > FACTOR: " << factor << endl;
	    }
	    else if ( token == "TRANSLATE") {
	      assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	    }
            else if ( token == "ROTATING") {
	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            } 
            else if ( token == "OSCILLATING") {
              osc_period = param->getDouble(iarg++)/(M_PI*2.0); // oscillation period
              
              if (mpi_rank == 0)
                cout << " > Oscillating wall: period: " << osc_period << endl; 
            }
            else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString()); 
	    }
	  }

          // Parse parameters for the boundary layer data exchanger

          Param * blde_param = getParam("BLDE");
          if ( blde_param) {
            iarg = 0;
            while ( iarg < blde_param->size()) {
              string token = blde_param->getString(iarg++);
              if (token == "NBL") {
                nbl = blde_param->getInt(iarg++);
              }
              else if ( token == "L_BL") {
                l_bl = blde_param->getDouble(iarg++);
              }
              else {
                CERR("unrecognized token " << token <<
                     " during parsing of BLDE in bc:\n" << param->str() << helpString());
              }
            }
          }
          if ( l_bl != BIG_DOUBLE && nbl != BIG_INT) {
            b_blde_params = true;

            registerData(bl_delta,zone_ptr->nbf,zone_ptr->name+":bl_delta",DEP_DATA);
            registerData(bl_delta_star,zone_ptr->nbf,zone_ptr->name+":bl_delta_star",DEP_DATA);
            registerData(bl_theta,zone_ptr->nbf,zone_ptr->name+":bl_theta",DEP_DATA);
            registerData(bl_pt_error,zone_ptr->nbf,zone_ptr->name+":bl_pt_error",DEP_DATA);
            registerData(bl_u_edge,zone_ptr->nbf,zone_ptr->name+":bl_u_e",DEP_DATA);
          }
	
          // define the bfocv behavior for this bc application.  it's defaulted
          // as true for hip and false for cuda, but we use the boolean to offer
          // the ability for either to toggle

#ifdef BUILD_CUDA

          b_use_bfocv = getBoolParam("USE_BFOCV", false);

#elif BUILD_HIP

          b_use_bfocv = getBoolParam("USE_BFOCV", false);

#else

          b_use_bfocv = false;  // wont get used at all in cpu land

#endif

        }
	
        ~WmAlgAdiabaticBc() {
	  
          DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);

          if (blde)
            delete blde;
	
        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		// y_plus is then...
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {

          // init boundary layer data exchanger
          
          if (b_blde_params) {
            assert(blde == NULL);
            blde = new BoundaryLayerDataExchanger(zone_ptr,nbl,l_bl);

            assert(bl_delta);
            assert(bl_delta_star);
            assert(bl_theta);
            assert(bl_pt_error);
            assert(bl_u_edge);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              bl_delta[ibf] = 0.0;
              bl_delta_star[ibf] = 0.0;
              bl_theta[ibf] = 0.0;
              bl_pt_error[ibf] = 0.0;
              bl_u_edge[ibf] = 0.0;
            }
          }

	  // init tau_wall...
	  
          assert(tau_wall);
	  
          if (!checkInit(tau_wall,zone_ptr->nbf)) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double u1 = MAG(u[icv]);
	      // set an initial value for tau_wall, this will constitute the
	      // initial guess for the solution when it is finally constructed.
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		tau_wall[ibf] = mu_lam[icv]*u1/y1;
	      }
	      else {
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  
          // init u_bc...
	  
          assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];
       	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)) {
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) { //TODO consider no "else" here to allow trans/rotate in staionary frame?
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( u_bc_d == NULL);
	  assert( tau_wall_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);

          if ( b_use_bfocv) { 

            if (mpi_rank == 0 ) 
              cout << " building bfocv for zone : " << zone_ptr->getName() << endl;
              
            zone_ptr->buildBfocvGpu();

          } 

#endif

	}

        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // since the wall model is relatively expensive, we just compute a 
	  // frozen value for the whole time step...
	  if (rk_stage == 1) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1    = zone_ptr->area_bf[ibf] / zone_ptr->area_over_delta_bf[ibf];
                double osc_fac = 1.0;
                if (osc_period > 0.0) {
                  osc_fac = sin(time/osc_period);
                }
                double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u[icv][i] - u_bc[ibf][i]*osc_fac;
		const double u1 = MAG(u_minus_ubc);
		// use the existing tau_wall to set a guess for the u_tau
		const double u_tau = sqrt(tau_wall[ibf] / rho[icv]);
		tau_wall[ibf] = AlgebraicWM::solve_tau(u1, y1, rho[icv], mu_lam[icv], u_tau);
		
	      } 
	      else {
		// assert(zone_ptr->area_bf[ibf] < 1.0E-16); //this should be a collapsed face.
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    //XXXXXXXXXXXXX should we use u or u parallel here??
            double osc_fac = 1.0;
            if (osc_period > 0.0) {
              osc_fac = sin(time/osc_period);
            }
	    double u_mag = 0.0;
	    for (int i =0; i < 3; ++i)
	      u_mag += (u[icv][i] - u_bc[ibf][i]*osc_fac) * (u[icv][i] - u_bc[ibf][i]*osc_fac);
	    u_mag = sqrt(max(0.0, u_mag));
	    double force[3];
	    if (u_mag > 0.0) {
	      FOR_I3 force[i] = factor*tau_wall[ibf]*(u[icv][i] - u_bc[ibf][i]*osc_fac)/u_mag*zone_ptr->area_bf[ibf];
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= force[i];
	      }
	      rhs[icv][4] -= DOT_PRODUCT(force, u_bc[ibf])*osc_fac;
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	  }
	}
	
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( !b_use_bfocv ) { 


            if ( zone_ptr->nbf > 0 ) { 

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

              LAUNCH_GPU_KERNEL_STREAM( addWmAlgAdiabaticFlux, grid_size, 
                                        block_size, stream, rhs, 
                                        sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                        mu_lam_d, u_bc_d, tau_wall_d,
                                        (cti_ffp) time,
                                        (cti_ffp) osc_period,
                                        zone_ptr->cvobf_d,
                                        zone_ptr->area_bf_d, 
                                        zone_ptr->area_over_delta_bf_d,
                                        zone_ptr->n_bf_d,
                                        zone_ptr->nbf);

            } 

          }
          else { 

            // avoid the use of atomic adds via a warp like structure 
            // that operates on bfocv.

            assert( zone_ptr->ncv_zone >= 0);

            if ( zone_ptr->ncv_zone > 0 ) { 

              assert( zone_ptr->bfocv_i_d);
              assert( zone_ptr->bfocv_v_d);
              assert( zone_ptr->cv_index_d);
         
              const int coop           = 16;
              const int repeat         = 1;
              const int block_size     = 128;
              const int grid_size      = 1 + (zone_ptr->ncv_zone *coop-1)/(repeat * block_size);

              LAUNCH_GPU_KERNEL_STREAM( (addWmAlgFluxCoop<cti_ffp,block_size,coop>),
                                        grid_size, block_size, stream, rhs,
                                        sp_vol_f_d, u_f_d, p_f_d, h_f_d, 
                                        mu_lam_d, u_bc_d, tau_wall_d,
                                        (cti_ffp) time,
                                        (cti_ffp) osc_period,
                                        zone_ptr->cv_index_d,
                                        zone_ptr->bfocv_i_d,
                                        zone_ptr->bfocv_v_d,
                                        zone_ptr->area_bf_d, 
                                        zone_ptr->area_over_delta_bf_d,
                                        zone_ptr->n_bf_d,
                                        zone_ptr->ncv_zone, repeat);

            }

          }

#endif

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {
          
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;

          if ( b_use_bfocv) 
            zone_ptr->destroyBfocvGpu();
#endif
        }
	
        void postLoadBalance() {
          initData();
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
#endif

          if (b_blde_params) {
            blde->computeBlFromPt(bl_delta,bl_delta_star,bl_theta,bl_u_edge,bl_thresh);
            blde->computeBlPtError(bl_pt_error);
          }

	}

        void query() {

          double my_buf[3] = {0.0,0.0,0.0};
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];

            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1  = zone_ptr->area_bf[ibf] / 
                                 zone_ptr->area_over_delta_bf[ibf];

              const double y_plus = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]          += zone_ptr->area_bf[ibf];
              my_buf[1]          += zone_ptr->area_bf[ibf]*tau_wall[ibf];
              my_buf[2]          += zone_ptr->area_bf[ibf]*y_plus;


            } 
          }

          double buf[3];
          MPI_Reduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:y_plus " << buf[2]/buf[0] 
                 << endl; 
          }
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          if ( zone_ptr->nbf > 0 ) { 

            // we need a temporary storage container on the device that is a 
            // vector nbf record... 

            ensureForceWorkArrayDevice(3*zone_ptr->nbf);
            assert( force_work_arr_d);
            cti::utils::deviceMemset(force_work_arr_d, 0, 3*zone_ptr->nbf*sizeof(cti_ffp));

            if ( (f_type == ForceType::TOTAL_FORCE   ) || 
                 ( f_type == ForceType::PRESSURE_FORCE)  ) { 

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size; 

              LAUNCH_GPU_KERNEL( addPressureForceDevice, grid_size, block_size, 
                                 force_work_arr_d, p_f_d, 
                                 zone_ptr->cvobf_d, zone_ptr->n_bf_d, 
                                 (cti_ffp) p_ref, zone_ptr->nbf);

            }

            if ( (f_type == ForceType::TOTAL_FORCE   ) || 
                 (f_type == ForceType::VISCOUS_FORCE))     { 


              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size; 

              LAUNCH_GPU_KERNEL( addViscousForceOscTauVarDevice, grid_size, block_size, 
                                 force_work_arr_d, u_f_d, tau_wall_d, u_bc_d,
                                 (cti_ffp) time, (cti_ffp) osc_period,
                                 zone_ptr->cvobf_d, zone_ptr->area_bf_d, 
                                 zone_ptr->nbf);


            } 

            cti::utils::copyToHost((double*)rval_dn3, force_work_arr_d, 3*zone_ptr->nbf);

          } 
#else 
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
 
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];// dp
            }
          }
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ){
            // tau_wall has already been computed at this point, but we'll assume
            // that the wall stress is instantaneously aligned with the velocity vector
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              
              double osc_fac = 1.0;
              if (osc_period > 0.0) {
                osc_fac = sin(time/osc_period);
              }
              //const double du[3] = DIFF(u[icv0],u_bc[ibf]);
              double du[3] = {u[icv0][0],u[icv0][1],u[icv0][2]};
              FOR_I3 du[i] -= osc_fac*u_bc[ibf][i];
              double u_mag = DOT_PRODUCT(du,du);
              if (u_mag > 0.0){
                u_mag = sqrt(max(0.0,u_mag));
                FOR_I3 rval_dn3[ibf][i] += tau_wall[ibf]*du[i]/u_mag*zone_ptr->area_bf[ibf]; 
              }
            }
          }
#endif
        }
      
      };
      
      
      class WmExchangeAdiabaticBc : public IdealGasBc {
      public:

	double * u_wall;
	double * xrot_wall;
	double * axisrot_wall;
	double omega_wall;

	bool b_stationary;

	double (*u_bc)[3];
	double *tau_wall;

        double *u_send;
        double *u_recv;
        
        cti_ffp * u_bc_d;
	cti_ffp * tau_wall_d;
        cti_ffp * u_send_d;
        cti_ffp * u_recv_d;

        int *ibf_local;
        int *ibf_local_d;
        int nbf_local;

        int n_disp; 

        DataExchanger * exchanger; 

	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WM_EXCHANGE_ADIABATIC [N = 2]"
	    "\n  [TRANSLATE <double> <double> <double>]"
	    "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
	    "\n  [STATIONARY]";
	}

	WmExchangeAdiabaticBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {

	  if (mpi_rank == 0) 
            cout << "WmExchangeAdiabaticBc: " << param->str() << endl;

          zone_ptr->b_wall  = true;
	  zone_ptr->lb_cost = 200;
	  
          u_bc         = NULL;
	  tau_wall     = NULL; 
          u_bc_d       = NULL;
	  tau_wall_d   = NULL;
	  
	  u_wall       = NULL;
	  xrot_wall    = NULL;
	  axisrot_wall = NULL;
	  omega_wall   = BIG_DOUBLE;

          u_recv       = NULL;
          u_recv_d     = NULL;
          u_send       = NULL;
          u_send_d     = NULL;

          exchanger    = NULL;
          ibf_local    = NULL;
          ibf_local_d  = NULL;
          nbf_local    = 0;

	  b_stationary = false;

          registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall", READ_DATA | WRITE_DATA);
	  registerFunc(&WmExchangeAdiabaticBc::y_plus,this,zone_ptr->name+":y_plus");

	  int iarg = 1;
          n_disp   = 2; 

	  while ( iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    
            if ( token == "TRANSLATE") {
	    
              assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	   
            } else if ( token == "ROTATING") {

	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            
            } else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      
              if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	  

            } else if ( (token == "N") || (token == "N_DISP")) { 

              n_disp = param->getInt(iarg++);


            } else {

	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString()); 
	    
            }
          }
	
        }
	
        ~WmExchangeAdiabaticBc() {
	  
          DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);

          DELETE(u_send);
          DELETE(u_recv);

          DELETE(ibf_local);

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( u_send_d ) { 

            cti::utils::freeOnDevice(u_send_d);
            u_send_d = NULL;

          } 

          if ( u_recv_d ) { 

            cti::utils::freeOnDevice(u_recv_d);
            u_recv_d = NULL;

          } 

          if ( u_bc_d ) { 

            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;

          } 

          if ( tau_wall_d) { 

            cti::utils::freeOnDevice(tau_wall_d);
            tau_wall_d = NULL;

          } 

          if ( ibf_local_d) { 

            cti::utils::freeOnDevice(ibf_local_d);
            ibf_local_d = NULL;

          } 

          if ( u_bc_d ) { 
            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;
          } 

          if ( tau_wall_d ) { 

            cti::utils::freeOnDevice(tau_wall_d); 
            tau_wall_d = NULL;

          } 
          

#endif

          if ( exchanger) { 
            delete exchanger;
          } 

	
        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		// y_plus is then...
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {

	  // init tau_wall...
	  
          assert(tau_wall);
	  
          if (!checkInit(tau_wall,zone_ptr->nbf)) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double u1 = MAG(u[icv]);
	      // set an initial value for tau_wall, this will constitute the
	      // initial guess for the solution when it is finally constructed.
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		tau_wall[ibf] = mu_lam[icv]*u1/y1;
	      }
	      else {
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  
          // init u_bc...
	  
          assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];
       	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)) {
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) { //TODO consider no "else" here to allow trans/rotate in staionary frame?
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( u_bc_d == NULL);
	  assert( tau_wall_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);

#endif

          // now we have to build the exchanger data structures ... 

          if ( mpi_rank == 0) { 
            cout << " > initializing exchanger for zone : " << zone_ptr->getName() << endl;
          } 

          assert ( exchanger == NULL); 


          double (*x_ex)[3] = new double[zone_ptr->nbf][3];
          double disp_fax   = 2.0*double(n_disp) - 1.0;  

          for ( int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const double delta = zone_ptr->area_bf[ibf] / zone_ptr->area_over_delta_bf[ibf];
            const double n_mag = MAG(zone_ptr->n_bf[ibf]);
            for (int i = 0; i < 3; ++i) 
              x_ex[ibf][i] = zone_ptr->x_bf[ibf][i] - zone_ptr->n_bf[ibf][i]/n_mag * delta * disp_fax; // inward pointing displacement ...  

          } 

          exchanger = new DataExchanger();
          exchanger->init(x_ex,zone_ptr->nbf);
         
          // at this point, it is possible that the exchanger did not find everyone
          // because some points are outside of the domain.   we need to find those
          // and treat them separately.

          assert( ibf_local == NULL);
          nbf_local = 0;

          int * bf_flag = new int[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            bf_flag[ibf] = -1;

          for (int ii =0; ii < exchanger->nx_unpack; ++ii) 
            bf_flag[exchanger->idopt_recv[ii]]++;

          // possible that we didnt find anyone, but check to make sure 
          // we dont have duplicates ... 

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)  
            assert( bf_flag[ibf] <= 0);
           
          for (int iter = 0; iter < 2; ++iter) { 

            nbf_local = 0;
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              if ( iter == 0) { 

                if ( bf_flag[ibf] == -1) 
                  ++nbf_local;

              } else { 

                if ( bf_flag[ibf] == -1) { 
                  ibf_local[nbf_local++] = ibf;
                }

              }
            }

            if ( iter == 0 ) { 

              assert( nbf_local == (zone_ptr->nbf - exchanger->nx_unpack));
              ibf_local = new int[nbf_local];

            } 
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( ibf_local_d == NULL);
          if ( nbf_local > 0 ) { 
            cti::utils::initAndCopyToDevice(ibf_local_d,ibf_local,nbf_local);
          }

#endif

          delete[] bf_flag;

          // we expect that the communicator is sparse... 
          
          exchanger->buildSparseCommunicatorStruct();
          const int n_recv = exchanger->recv_stuff.size()/3;
          //const int n_send = exchanger->send_stuff.size()/3;

          { 

            // actually lets check the worst case complexity of the 
            // communicator ... 

            int self_count = 0;
            for (int ii = 0; ii < n_recv; ++ii) { 

              int rank = exchanger->recv_stuff[3*ii];
              if ( rank == mpi_rank ) { 
                // this is the self send .. 
                self_count = exchanger->recv_stuff[3*ii+1];
                break;
              }
            }

            double my_fax = double(zone_ptr->nbf-self_count)/double(zone_ptr->nbf);
            double fax;
            MPI_Reduce(&my_fax,&fax,1,MPI_DOUBLE,MPI_MAX,0,mpi_comm);

            if ( mpi_rank == 0 ) 
              cout << " > worst case exch fraction : " << fax << endl;

          } 
         

          // lastly allocate some data for the vel magnitude .. 
        
          assert( u_send == NULL); u_send = new double[3*exchanger->nx_pack];
          assert( u_recv == NULL); u_recv = new double[3*exchanger->nx_unpack];

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( u_send_d == NULL);
          assert( u_recv_d == NULL);

          cti::utils::initOnDevice(u_send_d,3*exchanger->nx_pack);
          cti::utils::initOnDevice(u_recv_d,3*exchanger->nx_unpack);

#endif 

          delete[] x_ex;

	}

        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // since the wall model is relatively expensive, we just compute a 
	  // frozen value for the whole time step...
	  
          if (rk_stage == 1) {

            // we need to do the shuffle for the velocity field data now ... 
            // post the non-local sends and recvs .. 

            assert( exchanger);

            const int n_recv = exchanger->recv_stuff.size()/3;
            const int n_send = exchanger->send_stuff.size()/3;

            MPI_Request * recv_req = new MPI_Request[n_recv];
            MPI_Request * send_req = new MPI_Request[n_send];

            int jj_recv      = 0;
            int jj_send      = 0;

            const int UPDATE_TAG = 11112;

            int my_recv_disp  = -1;
            int my_self_count = -1;
            for (int ii = 0; ii < n_recv; ++ii) { 

              const int rank = exchanger->recv_stuff[3*ii];
              if ( rank != mpi_rank ) {
                
                MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                          3*exchanger->recv_stuff[3*ii+1], 
                          MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                          &recv_req[jj_recv]);

                ++jj_recv;

              } else { 

                // this is data i am sending to myself, so record 
                // the displacement 

                my_self_count = exchanger->recv_stuff[3*ii+1];
                my_recv_disp  = exchanger->recv_stuff[3*ii+2];

              }

            }


            // pack the velocity field that we need ... 

            for (int ii = 0; ii < exchanger->nx_pack; ++ii) { 

              const int icv      = exchanger->cvopt[ii];
              const double dx[3] = DIFF(exchanger->x_pack[ii],x_cv[icv]);

              double du[3];
              for (int i = 0; i < 3; ++i) { 
                du[i] = 0.0;
                for (int j = 0; j < 3; ++j) 
                  du[i] += dudx[icv][i][j] * dx[j];
              }


              for (int i = 0; i < 3; ++i) 
                u_send[3*ii+i] = u[icv][i] + du[i];

            } 


            // now post the sends

            for (int ii = 0; ii < n_send; ++ii) { 

              const int rank = exchanger->send_stuff[3*ii];
              
              if ( rank != mpi_rank) {

                MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2], 
                           3*exchanger->send_stuff[3*ii+1], 
                           MPI_DOUBLE,rank, 
                           UPDATE_TAG,mpi_comm,&send_req[jj_send]);

                ++jj_send;

              } else { 

                // check the counts for the self copy ... 

                assert( exchanger->send_stuff[3*ii+1] == my_self_count);
                const int send_disp = exchanger->send_stuff[3*ii+2];
                
                for (int kk = 0; kk < my_self_count; ++kk) {
                 
                  for (int i = 0; i < 3; ++i) 
                    u_recv[3*(my_recv_disp+kk)+i] = u_send[3*(send_disp+kk)+i];

                }

              }

            }

            // wait for the non-local recvs to complete 

            MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

	    for (int ii = 0; ii < exchanger->nx_unpack; ++ii) {

              const int ibf = exchanger->idopt_recv[ii];
	      
              const int icv = zone_ptr->cvobf[ibf];
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	
                const double disp_fax = 2.0*double(n_disp)-1.0;
                const double y1       = disp_fax * zone_ptr->area_bf[ibf] / 
                                        zone_ptr->area_over_delta_bf[ibf];
		
                double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u_recv[3*ii+i] - u_bc[ibf][i];
		const double u1 = MAG(u_minus_ubc);
		// use the existing tau_wall to set a guess for the u_tau
		const double u_tau = sqrt(tau_wall[ibf] / rho[icv]);
		tau_wall[ibf] = AlgebraicWM::solve_tau(u1, y1, rho[icv], mu_lam[icv], u_tau);
		
	      } 
	      else {
		// assert(zone_ptr->area_bf[ibf] < 1.0E-16); //this should be a collapsed face.
		tau_wall[ibf] = 0.0;
	      }
	    }
	
	    for (int ii = 0; ii < nbf_local; ++ii) {

              const int ibf = ibf_local[ii]; 
	      
              const int icv = zone_ptr->cvobf[ibf];
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	
                const double y1       = zone_ptr->area_bf[ibf] / 
                                        zone_ptr->area_over_delta_bf[ibf];
		
                double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u[icv][i] - u_bc[ibf][i];
		const double u1 = MAG(u_minus_ubc);
		// use the existing tau_wall to set a guess for the u_tau
		const double u_tau = sqrt(tau_wall[ibf] / rho[icv]);
		tau_wall[ibf] = AlgebraicWM::solve_tau(u1, y1, rho[icv], mu_lam[icv], u_tau);
		
	      } 
	      else {
		// assert(zone_ptr->area_bf[ibf] < 1.0E-16); //this should be a collapsed face.
		tau_wall[ibf] = 0.0;
	      }
	    }
	
            // wait for the sends to complete ...  

            MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

            delete[] send_req;
            delete[] recv_req; 

            // we're out.

          }

	  // add the flux... note that we are going in contig order for the vel recv.. 

          for (int ii = 0; ii < exchanger->nx_unpack; ++ii) { 

            const int ibf = exchanger->idopt_recv[ii];
	    const int icv = zone_ptr->cvobf[ibf];
	    
            // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    
            double u_mag = 0.0;
	    for (int i =0; i < 3; ++i) { 
              const double tmp = u_recv[3*ibf+i] - u_bc[ibf][i];
              u_mag           += tmp*tmp;
            } 
	    
            u_mag = sqrt(max(0.0, u_mag));
	    double force[3];
	    if (u_mag > 0.0) {
	      for (int i =0; i < 3; ++i) {
                force[i] = tau_wall[ibf]*(u_recv[3*ii+i] - u_bc[ibf][i])/u_mag
                           *zone_ptr->area_bf[ibf];
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= force[i];
	      }
	      rhs[icv][4] -= DOT_PRODUCT(force, u_bc[ibf]);
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	  }
	
          for (int ii = 0; ii < nbf_local; ++ii) { 

            const int ibf = ibf_local[ii]; 
	    const int icv = zone_ptr->cvobf[ibf];
	    
            // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    
            double u_mag = 0.0;
	    for (int i =0; i < 3; ++i) { 
              const double tmp = u[icv][i] - u_bc[ibf][i];
              u_mag           += tmp*tmp;
            } 
	    
            u_mag = sqrt(max(0.0, u_mag));
	    double force[3];
	    if (u_mag > 0.0) {
	      for (int i =0; i < 3; ++i) {
                force[i] = tau_wall[ibf]*(u[icv][i] - u_bc[ibf][i])/u_mag
                           *zone_ptr->area_bf[ibf];
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= force[i];
	      }
	      rhs[icv][4] -= DOT_PRODUCT(force, u_bc[ibf]);
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	  }

        }


        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)


          assert ( exchanger);


          if ( rk_stage == 1 ) { 


            const int n_recv = exchanger->recv_stuff.size()/3;
            const int n_send = exchanger->send_stuff.size()/3;

            MPI_Request * recv_req = new MPI_Request[n_recv];
            MPI_Request * send_req = new MPI_Request[n_send];

            int jj_recv      = 0;
            int jj_send      = 0;

            const int UPDATE_TAG = 11112;

            // post the receives first ... 

            int my_recv_disp  = -1;
            int my_self_count = -1;
            
            for (int ii = 0; ii < n_recv; ++ii) { 

              const int rank = exchanger->recv_stuff[3*ii];
              if ( rank != mpi_rank ) {
                
                MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                          3*exchanger->recv_stuff[3*ii+1], 
                          MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                          &recv_req[jj_recv]);

                ++jj_recv;

              } else { 

                // this is data i am sending to myself, so record 
                // the displacement 

                my_self_count = exchanger->recv_stuff[3*ii+1];
                my_recv_disp  = exchanger->recv_stuff[3*ii+2];

              }

            }


            // pack the data for the exchange ... 
            
            if ( exchanger->nx_pack > 0) { 

              const int block_size = 128;
              const int grid_size  = (exchanger->nx_pack + block_size - 1) / block_size;

              LAUNCH_GPU_KERNEL_STREAM( packVelocityForWmExchange, 
                                        grid_size, block_size, stream, 
                                        u_send_d,u_f_d,dudx_d,
                                        exchanger->cvopt_d,
                                        exchanger->x_pack_d,
                                        x_cv_d,exchanger->nx_pack);


            }

            // must synchronize this stream before we can copy the data back to 
            // host in order to facilitate the mpi communication... 

            DEVICE_SYNCHRONIZE_STREAM( stream );

            // XXX should consider using pinned memory for this transfer ...
            // i do not need to copy the whole send buffer bc some of the 
            // data is going to be local to this rank.... 

            int n_pack = 0;
            if ( n_send > 0 ) { 

              if ( exchanger->send_stuff[3*(n_send-1)] == mpi_rank ) { 

                // the send_disp is the count of the pack that isnt 
                // related to the self send ... 
                
                n_pack = exchanger->send_stuff[3*(n_send-1)+2]; 

              } else { 

                // i dont have a self send, so i need to whole buffer

                n_pack = exchanger->nx_pack;

              }
            }

            if ( n_pack > 0 ) { 
              cti::utils::copyToHost(u_send,u_send_d,3*n_pack);
            }

            // now post the sends

            for (int ii = 0; ii < n_send; ++ii) { 

              const int rank = exchanger->send_stuff[3*ii];
              
              if ( rank != mpi_rank) {

                MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2], 
                           3*exchanger->send_stuff[3*ii+1], 
                           MPI_DOUBLE,rank, 
                           UPDATE_TAG,mpi_comm,&send_req[jj_send]);

                ++jj_send;

              } else { 

                // check the counts for the self copy ... 

                assert( exchanger->send_stuff[3*ii+1] == my_self_count);

              }

            }


            // the communication is in flight right now.  while we wait, 
            // if there are any self communications that present then 
            // we can take of them here 

            if ( my_self_count > 0 ) {

              assert( n_send > 0);
              const int send_disp = exchanger->send_stuff[3*(n_send-1)+2];
              const int block_size = 128;
              const int grid_size  = (my_self_count + block_size - 1)/block_size;


              LAUNCH_GPU_KERNEL_STREAM( performWmExchangeSelfComm, 
                                        grid_size, block_size, stream, 
                                        u_recv_d, u_send_d, send_disp,
                                        my_recv_disp, my_self_count);


            }


            // now we have to wait ... 

            MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

           
            // we copy everything that isnt the self comm.  note 
            // that the self comm on the recv side is also at the end... 

            int n_unpack = 0;
            if ( n_recv > 0 ) { 

              if ( exchanger->recv_stuff[3*(n_recv-1)] == mpi_rank ) { 

                // the send_disp is the count of the pack that isnt 
                // related to the self send ... 
                
                n_unpack = exchanger->recv_stuff[3*(n_recv-1)+2]; 

              } else { 

                // i dont have a self send, so i need to whole buffer

                n_unpack = exchanger->nx_unpack;

              }
            }

            if ( n_unpack > 0 ) { 
              cti::utils::copyToDevice(u_recv_d,u_recv, 3*n_unpack);
            }


            // all the data is now there.  since we are performing 
            // the comm once per time step, we can compute the wall stress 

            if ( exchanger->nx_unpack > 0 ) { 

              const int block_size   = 128;
              const int grid_size    = (exchanger->nx_unpack + block_size - 1)/block_size;
              const cti_ffp disp_fax = 2.0*double(n_disp)-1.0;

              LAUNCH_GPU_KERNEL_STREAM( computeTauWallWmExchange, 
                                        grid_size, block_size, stream, 
                                        tau_wall_d,u_recv_d,u_bc_d, 
                                        sp_vol_f_d, mu_lam_d,
                                        exchanger->idopt_recv_d,
                                        zone_ptr->cvobf_d,
                                        zone_ptr->area_bf_d,
                                        zone_ptr->area_over_delta_bf_d,
                                        exchanger->nx_unpack,
                                        disp_fax);



            } 


            if (  nbf_local > 0 ) { 

              const int block_size   = 128;
              const int grid_size    = (nbf_local + block_size - 1)/block_size;

              LAUNCH_GPU_KERNEL_STREAM( computeTauWallWmNbfLocal, 
                                        grid_size, block_size, stream, 
                                        tau_wall_d,u_f_d,u_bc_d, 
                                        sp_vol_f_d, mu_lam_d,
                                        ibf_local_d,
                                        zone_ptr->cvobf_d,
                                        zone_ptr->area_bf_d,
                                        zone_ptr->area_over_delta_bf_d,
                                        nbf_local);

            } 


            // ensure the sends complete ... and we're out.

            MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

            delete[] send_req;
            delete[] recv_req; 

          }


          if ( exchanger->nx_unpack > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (exchanger->nx_unpack + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addWmAlgAdiabaticFluxExchange, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, u_bc_d, tau_wall_d, u_recv_d,
                                      exchanger->idopt_recv_d,
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      exchanger->nx_unpack);

          } 

          if ( nbf_local > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (nbf_local + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addWmAlgAdiabaticFluxLocal, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, u_bc_d, tau_wall_d, 
                                      ibf_local_d, 
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      nbf_local);



          } 
#endif

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {
          
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

          DELETE(ibf_local);
          nbf_local = 0;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;
          
          if ( ibf_local_d) { 
            cti::utils::freeOnDevice(ibf_local_d);
            ibf_local_d = NULL;
          }

#endif
        
          DELETE(u_send);
          DELETE(u_recv);
          DELETE(ibf_local);

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( u_send_d ) { 

            cti::utils::freeOnDevice(u_send_d);
            u_send_d = NULL;

          } 

          if ( u_recv_d ) { 

            cti::utils::freeOnDevice(u_recv_d);
            u_recv_d = NULL;

          } 

          if ( u_bc_d ) { 

            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;

          } 

          if ( tau_wall_d) { 

            cti::utils::freeOnDevice(tau_wall_d);
            tau_wall_d = NULL;

          } 

          if ( ibf_local_d) { 

            cti::utils::freeOnDevice(ibf_local_d);
            ibf_local_d = NULL;

          } 

#endif

          if ( exchanger) { 
            delete exchanger;
          } 



        }
	
        void postLoadBalance() {
          initData();
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
#endif
	}

        void query() {

          double my_buf[3] = {0.0,0.0,0.0};
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];

            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1  = zone_ptr->area_bf[ibf] / 
                                 zone_ptr->area_over_delta_bf[ibf];

              const double y_plus = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]          += zone_ptr->area_bf[ibf];
              my_buf[1]          += zone_ptr->area_bf[ibf]*tau_wall[ibf];
              my_buf[2]          += zone_ptr->area_bf[ibf]*y_plus;


            } 
          }

          double buf[3];
          MPI_Reduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:y_plus " << buf[2]/buf[0] 
                 << endl; 
          }
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          force_bf_tauvar_device(rval_dn3,f_type,this);
#else 
          force_bf_tauvar(rval_dn3,f_type,this);
#endif
        }
      };
      

      class WmAlgIsothermalBc : public IdealGasBc {
      private:

	double factor;

	double * u_wall;
	double * xrot_wall;
	double * axisrot_wall;
	double omega_wall;
        double osc_period;

	bool b_stationary;
        
        BoundaryLayerDataExchanger * blde;
        int nbl;
        double l_bl;
        bool b_blde_params;
        const double bl_thresh = 0.99;

        double * bl_delta;
        double * bl_delta_star;
        double * bl_theta;
        double * bl_pt_error;
        double * bl_u_edge;

      public:

	double T_wall;
	double (*u_bc)[3];
	double *tau_wall;
	double *q_wall;

        cti_ffp * u_bc_d;
	cti_ffp * tau_wall_d;
        cti_ffp * q_wall_d;
	
	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WM_ALG_ISOTHERMAL T_WALL <double>"
	    "\n  [FACTOR <double>]"
	    "\n  [TRANSLATE <double> <double> <double>]"
	    "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
	    "\n  [STATIONARY]";
	}
	
	WmAlgIsothermalBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {

	  if (mpi_rank == 0) cout << "WmAlgIsothermalBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  zone_ptr->lb_cost = 200;
	  
	  T_wall     = BIG_DOUBLE;
          u_bc       = NULL;
	  tau_wall   = NULL; 
	  q_wall     = NULL;
	  factor     = 1.0;

	  u_wall       = NULL;
	  xrot_wall    = NULL;
	  axisrot_wall = NULL;
	  omega_wall   = BIG_DOUBLE;
          osc_period   = -1.0;

	  b_stationary = false;

          u_bc_d     = NULL;
	  tau_wall_d = NULL;
          q_wall_d   = NULL;
          
          blde = NULL;
          nbl = BIG_INT;
          l_bl = BIG_DOUBLE;
          b_blde_params = false;

          bl_delta = NULL;
          bl_delta_star = NULL;
          bl_theta = NULL;
          bl_pt_error = NULL;
          bl_u_edge = NULL;
	  
          registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall",READ_DATA|WRITE_DATA);
          registerData(q_wall,zone_ptr->nbf,zone_ptr->name+":q_wall");
	  registerFunc(&WmAlgIsothermalBc::y_plus,this,zone_ptr->name+":y_plus");

	  int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if (token == "FACTOR") {
	      factor = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > FACTOR " << factor << endl;
	    }
	    else if ((token == "T")||(token == "T_WALL")) {
	      T_wall = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > T_WALL " << T_wall << endl;
	    }
	    else if ( token == "TRANSLATE") {
	      assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	    }
            else if ( token == "ROTATING") {
	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            } 
            else if ( token == "OSCILLATING") {
              osc_period = param->getDouble(iarg++)/(M_PI*2.0); // oscillation period
              
              if (mpi_rank == 0)
                cout << " > Oscillating wall: period: " << osc_period << endl;
            }
            else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString());
	    }
	  }

          // Parse parameters for the boundary layer data exchanger

          Param * blde_param = getParam("BLDE");
          if ( blde_param) {
            iarg = 0;
            while ( iarg < blde_param->size()) {
              string token = blde_param->getString(iarg++);
              if (token == "NBL") {
                nbl = blde_param->getInt(iarg++);
              }
              else if ( token == "L_BL") {
                l_bl = blde_param->getDouble(iarg++);
              }
              else {
                CERR("unrecognized token " << token <<
                     " during parsing of BLDE in bc:\n" << param->str() << helpString());
              }
            }
          }
          if ( l_bl != BIG_DOUBLE && nbl != BIG_INT) {
            b_blde_params = true;

            registerData(bl_delta,zone_ptr->nbf,zone_ptr->name+":bl_delta",DEP_DATA);
            registerData(bl_delta_star,zone_ptr->nbf,zone_ptr->name+":bl_delta_star",DEP_DATA);
            registerData(bl_theta,zone_ptr->nbf,zone_ptr->name+":bl_theta",DEP_DATA);
            registerData(bl_pt_error,zone_ptr->nbf,zone_ptr->name+":bl_pt_error",DEP_DATA);
            registerData(bl_u_edge,zone_ptr->nbf,zone_ptr->name+":bl_u_e",DEP_DATA);
          }
	  
	  if (T_wall == BIG_DOUBLE) {
	    CERR("missing T_WALL <double> during parsing of bc:\n" << 
		 param->str() << helpString());
	  }

	}
	
	~WmAlgIsothermalBc() {
	  DELETE(q_wall);
	  DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);

          if (blde)
            delete blde;

	}
	
	CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		// y_plus is then...
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {

          // init boundary layer data exchanger
          
          if (b_blde_params) {
            assert(blde == NULL);
            blde = new BoundaryLayerDataExchanger(zone_ptr,nbl,l_bl);

            assert(bl_delta);
            assert(bl_delta_star);
            assert(bl_theta);
            assert(bl_pt_error);
            assert(bl_u_edge);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              bl_delta[ibf] = 0.0;
              bl_delta_star[ibf] = 0.0;
              bl_theta[ibf] = 0.0;
              bl_pt_error[ibf] = 0.0;
              bl_u_edge[ibf] = 0.0;
            }
          }

	  // init tau_wall...
	  assert(tau_wall);
	  if (!checkInit(tau_wall,zone_ptr->nbf)) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double u1 = MAG(u[icv]);
	      // set an initial value for tau_wall, this will constitute the
	      // initial guess for the solution when it is finally constructed.
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		tau_wall[ibf] = mu_lam[icv]*u1/y1;
	      }
	      else {
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  // for q_wall we can just zero it...
	  assert(q_wall);
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    q_wall[ibf] = 0.0;
	  }
	  
          // init u_bc...
	  assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];	  
	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)) {
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) { //TODO consider no "else" here to allow trans/rotate in stationary frame?
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( u_bc_d     == NULL);
	  assert( tau_wall_d == NULL);
          assert( q_wall_d   == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(q_wall_d, q_wall, zone_ptr->nbf);

#endif

	}

        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // since the wall model is relatively expensive, we just compute a 
	  // frozen value for the whole time step...
	  if (rk_stage == 1) {
	    const double cp = R_gas*gamma/(gamma-1.0);
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		double osc_fac = 1.0;
                if (osc_period > 0.0) {
                  osc_fac = sin(time/osc_period);
                }
                
                //const double mag_u = MAG(u[icv]);
		//const double eps = mag_u*solver->dt / (y1 + solver->dt*mag_u);
		// \phi_new = eps*u + (1.0 -eps)*\phi_old
		// \phi_new - \phi_old = eps*(u - \phi_old)
		//u1[ibf]   += eps* (mag_u - u1[ibf]);
		//XXXXXXXXXXXXX should we use u or u parallel here??
		//const double u1 = MAG(u[icv]);  // Re 2000 channel test showed time averaging to have minimal impact, so removed
		double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u[icv][i] - u_bc[ibf][i]*osc_fac;
		const double u1 = MAG(u_minus_ubc);
		// use the existing tau_wall to set a guess for the u_tau
		const double u_tau = sqrt(tau_wall[ibf] / rho[icv]);
		tau_wall[ibf] = AlgebraicWM::solve_tau(u1, y1, rho[icv], mu_lam[icv], u_tau);
		q_wall[ibf]   = AlgebraicWM::compute_q_wall_approx(T[icv], T_wall, tau_wall[ibf],
								   rho[icv], mu_lam[icv],
								   loc_lam[icv], cp, y1);
	      }
	      else {
		// assert(zone_ptr->area_bf[ibf] < 1.0E-16); //this should be a collapsed face.
		tau_wall[ibf] = 0.0;
		q_wall[ibf] = 0.0;
	      }
	    }
	  }
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    //XXXXXXXXXXXXX should we use u or u parallel here??
            double osc_fac = 1.0;
            if (osc_period > 0.0) {
              osc_fac = sin(time/osc_period);
            }
	    double u_mag = 0.0;
	    for (int i =0; i < 3; ++i)
	      u_mag += (u[icv][i] - u_bc[ibf][i]*osc_fac) * (u[icv][i] - u_bc[ibf][i]*osc_fac);
	    u_mag = sqrt(max(0.0, u_mag));
	    double force[3];
	    if (u_mag > 0.0) {
	      FOR_I3 force[i] = factor*tau_wall[ibf]*(u[icv][i] - u_bc[ibf][i]*osc_fac)/u_mag*zone_ptr->area_bf[ibf];
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= force[i];
	      }
	      rhs[icv][4] -= DOT_PRODUCT(force, u_bc[ibf])*osc_fac;
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	    rhs[icv][4] -= q_wall[ibf]*zone_ptr->area_bf[ibf];
	  }
	}
	
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;
            const double cp      = R_gas * gamma / (gamma- 1.0);

            LAUNCH_GPU_KERNEL_STREAM( addWmAlgIsothermalFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, loc_lam_d, u_bc_d, 
                                      tau_wall_d, q_wall_d,
                                      (cti_ffp) time,
                                      (cti_ffp) osc_period,
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      zone_ptr->nbf, 
                                      (cti_ffp) T_wall, (cti_ffp) cp);

          }

#endif

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {

	  // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d);     u_bc_d = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;
          cti::utils::freeOnDevice(q_wall_d);   q_wall_d   = NULL;
#endif
        }
	
        void postLoadBalance() {
          initData();
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
          cti::utils::copyToHost(q_wall, q_wall_d,zone_ptr->nbf);
#endif

          if (b_blde_params) {
            blde->computeBlFromPt(bl_delta,bl_delta_star,bl_theta,bl_u_edge,bl_thresh);
            blde->computeBlPtError(bl_pt_error);
          }

	}

        void query() {
          
          double my_buf[4] = {0.0,0.0,0.0,0.0};

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];
            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1       = zone_ptr->area_bf[ibf] / 
                                      zone_ptr->area_over_delta_bf[ibf];
              const double y_plus   = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]            += zone_ptr->area_bf[ibf];
              my_buf[1]            += zone_ptr->area_bf[ibf] * tau_wall[ibf];
              my_buf[2]            += zone_ptr->area_bf[ibf] * y_plus;
              my_buf[3]            += zone_ptr->area_bf[ibf] * q_wall[ibf];

            } 
          } 

          double buf[4];
          MPI_Reduce(my_buf,buf,4,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:avg(y_plus) " << buf[2]/buf[0] 
                 << " 12:int(q_wall)dA " << buf[3] 
		 << " 14:area " << buf[0]
		 << endl; 
          
          }
        }
        
        void force(double (*rval_dn3)[3], const ForceType f_type) { 
          
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          if ( zone_ptr->nbf > 0 ) { 

            // we need a temporary storage container on the device that is a 
            // vector nbf record... 

            ensureForceWorkArrayDevice(3*zone_ptr->nbf);
            assert( force_work_arr_d);
            cti::utils::deviceMemset(force_work_arr_d, 0, 3*zone_ptr->nbf*sizeof(cti_ffp));

            if ( (f_type == ForceType::TOTAL_FORCE   ) || 
                 ( f_type == ForceType::PRESSURE_FORCE)  ) { 

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size; 

              LAUNCH_GPU_KERNEL( addPressureForceDevice, grid_size, block_size, 
                                 force_work_arr_d, p_f_d, 
                                 zone_ptr->cvobf_d, zone_ptr->n_bf_d, 
                                 (cti_ffp) p_ref, zone_ptr->nbf);

            }

            if ( (f_type == ForceType::TOTAL_FORCE   ) || 
                 (f_type == ForceType::VISCOUS_FORCE))     { 


              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size; 

              LAUNCH_GPU_KERNEL( addViscousForceOscTauVarDevice, grid_size, block_size, 
                                 force_work_arr_d, u_f_d, tau_wall_d, u_bc_d,
                                 (cti_ffp) time, (cti_ffp) osc_period,
                                 zone_ptr->cvobf_d, zone_ptr->area_bf_d, 
                                 zone_ptr->nbf);


            } 

            cti::utils::copyToHost((double*)rval_dn3, force_work_arr_d, 3*zone_ptr->nbf);

          } 
#else 
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
 
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];// dp
            }
          }
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ){
            // tau_wall has already been computed at this point, but we'll assume
            // that the wall stress is instantaneously aligned with the velocity vector
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              
              double osc_fac = 1.0;
              if (osc_period > 0.0) {
                osc_fac = sin(time/osc_period);
              }
              //const double du[3] = DIFF(u[icv0],u_bc[ibf]);
              double du[3] = {u[icv0][0],u[icv0][1],u[icv0][2]};
              FOR_I3 du[i] -= osc_fac*u_bc[ibf][i];
              double u_mag = DOT_PRODUCT(du,du);
              if (u_mag > 0.0){
                u_mag = sqrt(max(0.0,u_mag));
                FOR_I3 rval_dn3[ibf][i] += tau_wall[ibf]*du[i]/u_mag*zone_ptr->area_bf[ibf]; 
              }
            }
          }
#endif
        }

      };

      class WmAlgChtBc : public IdealGasBc {
      public:

	double *T_wall;
	double *tau_wall;
	double *q_wall;
        double (*u_bc)[3];
	double factor;
	double *r_wall; // used for the search radius for interface connection

	cti_ffp * tau_wall_d;
        cti_ffp * T_wall_d;
        cti_ffp * q_wall_d;
        cti_ffp * u_bc_d;
	
	string helpString() {
	  return 
	    "\nExamples:"
	    "\n  <zonename> WM_ALG_CHT";
	}
	
	WmAlgChtBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {

	  if (mpi_rank == 0) cout << "WmAlgChtBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  assert(zone_ptr->interface == 0);
	  zone_ptr->interface = CHT_INTERFACE;
	  zone_ptr->lb_cost = 200;
	  
	  T_wall     = NULL;
	  tau_wall   = NULL; 
	  q_wall     = NULL;
          u_bc       = NULL;
	  factor     = 1.0;
	  r_wall     = NULL;
	  
	  tau_wall_d = NULL;
          T_wall_d   = NULL;
          q_wall_d   = NULL;
          u_bc_d     = NULL;
	  
          registerData(T_wall,zone_ptr->nbf,zone_ptr->name+":T_wall");
          registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall",READ_DATA|WRITE_DATA);
          registerData(q_wall,zone_ptr->nbf,zone_ptr->name+":q_wall");
	  registerFunc(&WmAlgChtBc::y_plus,this,zone_ptr->name+":y_plus");

          registerData(r_wall,zone_ptr->nbf,zone_ptr->name+":r_wall",READ_DATA|WRITE_DATA);

	  int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if (token == "FACTOR") {
	      factor = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > FACTOR " << factor << endl;
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString());
	    }
	  }
	  
	}
	
	~WmAlgChtBc() {
	  DELETE(q_wall); 
	  DELETE(tau_wall);
	  DELETE(T_wall);
          DELETE(u_bc);
	  DELETE(r_wall);
	}
	
	CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		// y_plus is then...
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

	void initData() {

	  // init tau_wall...
	  assert(tau_wall);
	  if (!checkInit(tau_wall,zone_ptr->nbf)) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double u1 = MAG(u[icv]);
	      // set an initial value for tau_wall, this will constitute the
	      // initial guess for the solution when it is finally constructed.
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		tau_wall[ibf] = mu_lam[icv]*u1/y1;
	      }
	      else {
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  // for q_wall,T_wall we can just zero it...
	  assert(T_wall);
	  assert(q_wall);
          assert(u_bc == NULL); 
          u_bc = new double[zone_ptr->nbf][3];	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    T_wall[ibf]  = 0.0;
	    q_wall[ibf]  = 0.0;
	    FOR_I3 u_bc[ibf][i] = 0.0;
	  }

	  // r_wall is initialized during the init of cht. It cannot be done here,
	  // because this is currently TOO LATE. all we can do here is check that it 
	  // has been done...
	  assert(checkInit(r_wall,zone_ptr->nbf));
	  
#if defined(BUILD_CUDA) || defined(BUILD_HIP)

	  assert( tau_wall_d == NULL);
          assert( T_wall_d   == NULL); 
          assert( q_wall_d   == NULL);
          assert( u_bc_d     == NULL);

          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(T_wall_d,T_wall,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(q_wall_d,q_wall,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_bc_d  , (double*) u_bc, 3*zone_ptr->nbf);

          zone_ptr->buildDaobfGpu();
#endif

	}

        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // since the wall model is relatively expensive, we just compute a 
	  // frozen value for the whole time step...
	  if (rk_stage == 1) {
	    const double cp = R_gas*gamma/(gamma-1.0);
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      // reconstruct T_wall from the cht_T_fluid_buf...
	      // recall that zone_ptr->daobf_wgt sums to 1.0 for each bf...
	      T_wall[ibf] = 0.0;
	      for (int dob = zone_ptr->daobf_i[ibf]; dob != zone_ptr->daobf_i[ibf+1]; ++dob) {
		const int ida = zone_ptr->daobf_v[dob];
		T_wall[ibf] += zone_ptr->daobf_wgt[dob]*cht_T_fluid_buf[ida];
	      }
	      // now compute the wall stress and flux based on this T_wall
	      const int icv = zone_ptr->cvobf[ibf];
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const double u1 = MAG(u[icv]);
		// use the existing tau_wall to set a guess for the u_tau
		const double u_tau = sqrt(tau_wall[ibf] / rho[icv]);
		tau_wall[ibf] = AlgebraicWM::solve_tau(u1, y1, rho[icv], mu_lam[icv], u_tau);
		q_wall[ibf]   = AlgebraicWM::compute_q_wall_approx(T[icv], T_wall[ibf], tau_wall[ibf],
								   rho[icv], mu_lam[icv],
								   loc_lam[icv], cp, y1);
	      }
	      else {
		// assert(zone_ptr->area_bf[ibf] < 1.0E-16); //this should be a collapsed face.
		tau_wall[ibf] = 0.0;
		q_wall[ibf] = 0.0;
	      }
	    }
	  }
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    // any need to reconstruct T_wall in each timestep? for now, no...
	    const int icv = zone_ptr->cvobf[ibf];
	    // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    //XXXXXXXXXXXXX should we use u or u parallel here??
	    double u_mag = MAG(u[icv]);
	    if (u_mag > 0.0) {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= factor*tau_wall[ibf]*u[icv][i]/u_mag*zone_ptr->area_bf[ibf];
	      }
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	    const double Q = q_wall[ibf]*zone_ptr->area_bf[ibf];
	    rhs[icv][4] -= Q;
	    // send Q back, split among the cht data...
	    for (int dob = zone_ptr->daobf_i[ibf]; dob != zone_ptr->daobf_i[ibf+1]; ++dob) {
	      const int ida = zone_ptr->daobf_v[dob];
	      cht_q_fluid_buf[ida] += zone_ptr->daobf_wgt[dob]*Q;
	    }
	  }
	}
	
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( (rk_stage == 1) && (zone_ptr->nbf > 0 ) ) { 

            // compute tau_wall, q_wall and reduce T_wall .. 
            // assumes that cht_T_fluid_buf_d is properly populated on the host 
            // at this point ... 

            const cti_ffp cp     = R_gas*gamma/(gamma-1.0f);
            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( computeWmChtStuff, grid_size, block_size, stream, 
                                      tau_wall_d,q_wall_d,T_wall_d,
                                      sp_vol_f_d, u_f_d, h_f_d,
                                      mu_lam_d, loc_lam_d,
                                      zone_ptr->daobf_i_d,zone_ptr->daobf_v_d,
                                      zone_ptr->daobf_wgt_d,cht_T_fluid_buf_d,
                                      zone_ptr->area_bf_d,zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->cvobf_d, cp,zone_ptr->nbf);


          } 


          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;
	    
            LAUNCH_GPU_KERNEL_STREAM( addWmChtFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      mu_lam_d, 
                                      tau_wall_d, q_wall_d,
                                      zone_ptr->cvobf_d,
                                      zone_ptr->area_bf_d, 
                                      zone_ptr->area_over_delta_bf_d,
                                      zone_ptr->n_bf_d,
                                      zone_ptr->daobf_i_d,zone_ptr->daobf_v_d,
                                      zone_ptr->daobf_wgt_d,cht_q_fluid_buf_d,
                                      zone_ptr->nbf);

          }

#endif

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {

	  // we could register u_bc and not have to worry about manually
	  // handling this...

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;
          cti::utils::freeOnDevice(T_wall_d);   T_wall_d   = NULL;
          cti::utils::freeOnDevice(q_wall_d);   q_wall_d   = NULL;

          zone_ptr->destroyDaobfGpu();

#endif
        }
	
        void postLoadBalance() {
          initData();
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
          cti::utils::copyToHost(T_wall,T_wall_d,zone_ptr->nbf);
          cti::utils::copyToHost(q_wall,q_wall_d,zone_ptr->nbf);
#endif
	}

        void query() {

	  double my_buf[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	  double my_minmax[4] = { BIG_DOUBLE, BIG_DOUBLE, BIG_DOUBLE, BIG_DOUBLE };
	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      const double y1     = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
	      const int icv       = zone_ptr->cvobf[ibf];
	      const double u_tau  = sqrt(tau_wall[ibf]/rho[icv]);
	      const double nu     = mu_lam[icv]/rho[icv];
	      const double y_plus = y1*u_tau/nu;
	      my_buf[0] += zone_ptr->area_bf[ibf];
	      my_buf[1] += zone_ptr->area_bf[ibf]*tau_wall[ibf];
	      my_buf[2] += zone_ptr->area_bf[ibf]*y_plus;
	      my_buf[3] += zone_ptr->area_bf[ibf]*q_wall[ibf];
	      my_buf[4] += zone_ptr->area_bf[ibf]*T_wall[ibf];
	      my_minmax[0] = min(my_minmax[0],T_wall[ibf]);
	      my_minmax[1] = min(my_minmax[1],-T_wall[ibf]);
	      my_buf[5] += zone_ptr->area_bf[ibf]*T[icv];
	      my_minmax[2] = min(my_minmax[2],T[icv]);
	      my_minmax[3] = min(my_minmax[3],-T[icv]);
	    }
	  }
	  
	  double buf[6];
	  MPI_Reduce(my_buf,buf,6,MPI_DOUBLE,MPI_SUM,0,mpi_comm);
	  double minmax[4];
	  MPI_Reduce(my_minmax,minmax,4,MPI_DOUBLE,MPI_MIN,0,mpi_comm);
  
	  if ( mpi_rank == 0 ) {
	    
            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
	         << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:avg(y_plus) " << buf[2]/buf[0] 
	         << " 12:int(q_wall)dA " << buf[3] 
	         << " 14:area " << buf[0] 
	         << " 16-18:Twall(min,avg,max) " << minmax[0] << " " << buf[4]/buf[0] << " " << -minmax[1] 
                 << " 20-22:proj(T)(min,avg,max) " << minmax[2] << " " << buf[5]/buf[0] << " " << -minmax[3] 
                 << endl;
	  }
	  
	}
     
        void force(double (*rval_dn3)[3], const ForceType f_type) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          force_bf_tauvar_device(rval_dn3,f_type,this);
#else 
          force_bf_tauvar(rval_dn3,f_type,this);
#endif
        }

      };

      class WmSlipIsothermalBc : public IdealGasBc {
      public:
	double *cdel_w;
	double (*rhou_s)[3];
        double (*u_wall)[3];
	double *T_s;
        //double (*u_bc)[3];
	double T_bc;
	double c_slip;

        double *mean_rhoun_s;
        double t_avg;
        double time_init;

        bool b_sgs_slip;
        double *sgs_cdel_w;
        double *sgs_f_w;
        double (*sij_w)[6];
        double (*betaij_w)[6];
        double *BS;
        double *BB;
        double *BR;

        double (*u_avg)[3];
        double *uu_avg;
        double *utau_avg;

        double (*tau_wall_)[3];

        cti_ffp * cdel_w_d;
        cti_ffp * rhou_s_d; 
        cti_ffp * T_s_d;
        cti_ffp * u_wall_d;

        cti_ffp * mean_rhoun_s_d;

        cti_ffp * sgs_cdel_w_d;
        cti_ffp * sgs_f_w_d;
        cti_ffp * sij_w_d;
        cti_ffp * betaij_w_d;
        cti_ffp * BS_d;
        cti_ffp * BB_d;
        cti_ffp * BR_d;

        cti_ffp * u_avg_d;
        cti_ffp * uu_avg_d;
        cti_ffp * utau_avg_d;

        cti_ffp * tau_wall_d_;

        bool b_param1;
        bool b_param2;
        bool b_param3;
        bool b_nodamp;
        bool b_lam_sensor;

        bool b_adiabatic;

        BoundaryLayerDataExchanger * blde;
        int nbl;
        double l_bl;
        bool b_blde_params;
        const double bl_thresh = 0.99;

        double * bl_delta;
        double * bl_delta_star;
        double * bl_theta;
        double * bl_u_edge;

        double * res_d;

        WmSlipIsothermalBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  
          if (mpi_rank == 0) 
            cout << "WmSlipIsothermalBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
          zone_ptr->lb_cost = 50;
	  cdel_w = NULL;
	  rhou_s = NULL;
	  
          T_s    = NULL;
          u_wall = NULL;

          mean_rhoun_s = NULL;
          t_avg = 0.0;
          time_init = 0.0;

          cdel_w_d = NULL;
          rhou_s_d = NULL;
          T_s_d    = NULL;
          u_wall_d = NULL;

          mean_rhoun_s_d = NULL;

          b_sgs_slip = false;
          sgs_cdel_w = NULL;
          sgs_f_w    = NULL;
          sij_w      = NULL;
          betaij_w   = NULL;
          BS         = NULL;
          BB         = NULL;
          BR         = NULL;

          u_avg      = NULL;
          uu_avg     = NULL;
          utau_avg   = NULL;

          tau_wall_ = NULL;
          tau_wall_d_ = NULL;

          sgs_cdel_w_d = NULL;
          sgs_f_w_d    = NULL;
          sij_w_d      = NULL;
          betaij_w_d   = NULL;
          BS_d         = NULL;
          BB_d         = NULL;
          BR_d         = NULL;
        
          u_avg_d    = NULL;
          uu_avg_d   = NULL;
          utau_avg_d = NULL;

          b_param1 = false;
          b_param2 = false;
          b_param3 = false;
          b_nodamp = false;
          b_lam_sensor = false;

          b_adiabatic = false;

          blde = NULL;
          nbl = BIG_INT;
          l_bl = BIG_DOUBLE;
          b_blde_params = false;

          bl_delta = NULL;
          bl_delta_star = NULL;
          bl_theta = NULL;
          bl_u_edge = NULL;

          res_d = NULL;
	  
          //registerFunc(&WmSlipIsothermalBc::tau_wall,this,zone_ptr->name+":tau_wall");
	  registerFunc(&WmSlipIsothermalBc::y_plus,this,zone_ptr->name+":y_plus");
          registerFunc(&WmSlipIsothermalBc::cslip_func,this,zone_ptr->name+":cslip");

          registerFunc(&WmSlipIsothermalBc::rhous_func,this,zone_ptr->name+":rhous");

          registerData(mean_rhoun_s,zone_ptr->nbf,zone_ptr->name+":mean_rhoun_s", READ_DATA|WRITE_DATA);

          registerData(u_avg,zone_ptr->nbf,zone_ptr->name+":u_avg", READ_DATA|WRITE_DATA);
          registerData(uu_avg,zone_ptr->nbf,zone_ptr->name+":uu_avg", READ_DATA|WRITE_DATA);
          registerData(utau_avg,zone_ptr->nbf,zone_ptr->name+":utau_avg", READ_DATA|WRITE_DATA);
	  
          bool b_T_bc = false;

          c_slip      = 0.41;
	  
          int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ((token == "T")||(token == "T_WALL")||(token == "T_BC")) {
	      T_bc = param->getDouble(iarg++);
	      b_T_bc = true;
	    }
	    else if (token == "C_SLIP") {
	      c_slip = param->getDouble(iarg++);
	    }
            else if (token == "SGS_SLIP") {
              b_sgs_slip = true;
            }
            else if ((token == "DT_AVG")||(token == "T_AVG")||(token == "TIME_AVG")) {
              t_avg = param->getDouble(iarg++);
            }
            else if (token == "NO_DAMP") {
              b_nodamp = true;
            }
            else if (token == "PARAM1") {
              b_param1 = true;
            }
            else if (token == "PARAM2") {
              b_param2 = true;
            }
            else if (token == "PARAM3") {
              b_param3 = true;
            }
            else if (token == "LAMINAR_SENSOR") {
              b_lam_sensor = true;
            }
            else if (token == "ADIABATIC") {
              b_adiabatic = true;
            }
	    else {
	      CERR("unrecognized WM_SLIP token: " << token);
	    }
	  }
	  //int ierr = 0;
	  if (!b_T_bc) {
	    CERR("WM_SLIP requires params: T <double> C_SLIP [<double> = 0.41]");
	  }
          
          // Parse parameters for the boundary layer data exchanger

          Param * blde_param = getParam("BLDE");
          if ( blde_param) {
            iarg = 0;
            while ( iarg < blde_param->size()) {
              string token = blde_param->getString(iarg++);
              if (token == "NBL") {
                nbl = blde_param->getInt(iarg++);
              }
              else if ( token == "L_BL") {
                l_bl = blde_param->getDouble(iarg++);
              }
              else {
                CERR("unrecognized token " << token <<
                     " during parsing of BLDE in bc:\n" << param->str());
              }
            }
          }
          if ( l_bl != BIG_DOUBLE && nbl != BIG_INT) {
            b_blde_params = true;

            registerData(bl_delta,zone_ptr->nbf,zone_ptr->name+":bl_delta",DEP_DATA);
            registerData(bl_delta_star,zone_ptr->nbf,zone_ptr->name+":bl_delta_star",DEP_DATA);
            registerData(bl_theta,zone_ptr->nbf,zone_ptr->name+":bl_theta",DEP_DATA);
            registerData(bl_u_edge,zone_ptr->nbf,zone_ptr->name+":bl_u_e",DEP_DATA);
          }

	}
	
        ~WmSlipIsothermalBc() {
	  
          DELETE(cdel_w);
	  DELETE(rhou_s);
	  DELETE(T_s);
          DELETE(u_wall);

          DELETE(mean_rhoun_s);

          if (b_sgs_slip) {
            DELETE(sgs_cdel_w);
            DELETE(sgs_f_w);
            DELETE(sij_w);
            DELETE(betaij_w);
            DELETE(BS);
            DELETE(BB);
            DELETE(BR);
          }

          DELETE(u_avg);
          DELETE(uu_avg);
          DELETE(utau_avg);

          if (b_blde_params) {
            DELETE(bl_delta);
            DELETE(bl_delta_star);
            DELETE(bl_theta);
            DELETE(bl_u_edge);
            delete blde;
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( mean_rhoun_s_d) {
            cti::utils::freeOnDevice(mean_rhoun_s_d);
            mean_rhoun_s_d = NULL;
          }

          if ( sgs_cdel_w_d) {
            cti::utils::freeOnDevice(sgs_cdel_w_d);
            sgs_cdel_w_d = NULL;
          }
          if ( sgs_f_w_d) {
            cti::utils::freeOnDevice(sgs_f_w_d);
            sgs_f_w_d = NULL;
          }
          if ( sij_w_d) {
            cti::utils::freeOnDevice(sij_w_d);
            sij_w_d = NULL;
          }
          if ( betaij_w_d) {
            cti::utils::freeOnDevice(betaij_w_d);
            betaij_w_d = NULL;
          }
          if ( BS_d) {
            cti::utils::freeOnDevice(BS_d);
            BS_d = NULL;
          }
          if ( BB_d) {
            cti::utils::freeOnDevice(BB_d);
            BB_d = NULL;
          }
          if ( BR_d) {
            cti::utils::freeOnDevice(BR_d);
            BR_d = NULL;
          }
          if ( u_avg_d) {
            cti::utils::freeOnDevice(u_avg_d);
            u_avg_d = NULL;
          }
          if ( uu_avg_d) {
            cti::utils::freeOnDevice(uu_avg_d);
            uu_avg_d = NULL;
          }
          if ( utau_avg_d) {
            cti::utils::freeOnDevice(utau_avg_d);
            utau_avg_d = NULL;
          }

          if ( res_d) { 
            cti::utils::freeOnDevice(res_d); res_d = NULL;
          }

#endif
	
        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {

          // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		
                // TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		
                // y_plus is then...
		rval_dn[ibf] = y1*sqrt(MAG(tau_wall_[ibf])*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}

        CtiData *cslip_func(vector<CtiData*>& argVec,const bool b_eval) {

          if (!argVec.empty()) {
            eval_cerr << "cslip() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {

                const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
                rval_dn[ibf] = cdel_w[ibf]/y1;

              }
              else {
                rval_dn[ibf] = 0.0;
              }
            }
          }
          return rval;
        }
	
        CtiData *tau_wall(vector<CtiData*>& argVec,const bool b_eval) {
	  
          if (!argVec.empty()) {
	    eval_cerr << "tau_wall() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  
          if (b_eval) { 
	  
            double *rval_dn = rval->dn();
	    
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

              rval_dn[ibf] = MAG(tau_wall_[ibf]); 
              
            }
            
          }
	
          return rval;
	}

        CtiData *rhous_func(vector<CtiData*>& argVec,const bool b_eval) {

          int comp = -1;
          if (argVec.size() == 1) {
            if (argVec[0]->getDatatype() != I_DATA) {
              eval_cerr << "rhous() argument expects I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
              return NULL;
            }
            comp = argVec[0]->i();
            if (( comp < 0) || (comp >= 3) ) {
              eval_cerr << "rhous() integer argument should be in [0:3), got " << comp << endl;
              return NULL;
            }
          }
          else if (argVec.size() > 1) {
            eval_cerr << "tau_wall() accepts up to 1 argument: got " << argVec.size() << endl;
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) {

            double *rval_dn = rval->dn();
            if (comp == -1) {
              for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
                rval_dn[ibf] = MAG(rhou_s[ibf]);
              }
            }
            else {
              assert((comp >= 0)&&(comp < 3));
              for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
                rval_dn[ibf] = rhou_s[ibf][comp];
              }
            }
          }
          return rval;
        }
	
        void initData() {

          // init boundary layer data exchanger
          
          if (b_blde_params) {
            assert(blde == NULL);
            blde = new BoundaryLayerDataExchanger(zone_ptr,nbl,l_bl);

            assert(bl_delta);
            assert(bl_delta_star);
            assert(bl_theta);
            assert(bl_u_edge);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              bl_delta[ibf] = 0.0;
              bl_delta_star[ibf] = 0.0;
              bl_theta[ibf] = 0.0;
              bl_u_edge[ibf] = 0.0;
            }
          }

	  cdel_w = new double[zone_ptr->nbf];
	  rhou_s = new double[zone_ptr->nbf][3];
	  T_s    = new double[zone_ptr->nbf];
	 
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
	      cdel_w[ibf] = c_slip*delta;
	    }
	    else {
	      cdel_w[ibf] = 0.0;
	    }
            //FOR_I3 u_bc[ibf][i] = 0.0;
	  }

          if (b_sgs_slip) {
            sgs_cdel_w = new double[zone_ptr->nbf];
            sgs_f_w = new double[zone_ptr->nbf];
            sij_w = new double[zone_ptr->nbf][6];
            betaij_w = new double[zone_ptr->nbf][6];
            BS = new double[zone_ptr->nbf];
            BB = new double[zone_ptr->nbf];
            BR = new double[zone_ptr->nbf];
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              sgs_cdel_w[ibf] = 0.0;
              sgs_f_w[ibf] = 0.0;
              for (int i = 0; i < 6; ++i) {
                sij_w[ibf][i] = 0.0;
                betaij_w[ibf][i] = 0.0;
              }
              BS[ibf] = 0.0;
              BB[ibf] = 0.0;
              BR[ibf] = 0.0;
            }
          }

          assert( tau_wall_ == NULL);
          tau_wall_ = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            FOR_I3 tau_wall_[ibf][i] = 0.0;

          assert(mean_rhoun_s);
          if ( !checkInit(mean_rhoun_s,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_rhoun_s[ibf] = 0.0;
            }
            time_init = time;
          }

          assert(u_avg);
          if ( !checkInit(u_avg,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 u_avg[ibf][i] = 0.0;
            }
          }

          assert(uu_avg);
          if ( !checkInit(uu_avg,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              uu_avg[ibf] = 0.0;
            }
          }

          assert(utau_avg);
          if ( !checkInit(utau_avg,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              utau_avg[ibf] = 0.0;
            }
          }

          // place holder for parsing STATIONARY, etc for the velocity ..XXX

          u_wall = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            for (int i = 0; i < 3; ++i) 
              u_wall[ibf][i] = 0.0;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( rhou_s_d == NULL);
          assert( T_s_d    == NULL);
          assert( cdel_w_d == NULL);
          assert( u_wall_d == NULL);
          assert( tau_wall_d_ == NULL);

          cti::utils::initOnDevice(rhou_s_d,3*zone_ptr->nbf);
          cti::utils::initOnDevice(T_s_d   ,  zone_ptr->nbf);
          
          cti::utils::initAndCopyToDevice(tau_wall_d_,(double*)tau_wall_,3*zone_ptr->nbf);

          // set an initial guess for the cdel_w .. will be updated
          cti::utils::initAndCopyToDevice(cdel_w_d,cdel_w,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_wall_d,(double*)u_wall,3*zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(mean_rhoun_s_d,mean_rhoun_s,zone_ptr->nbf);

          if (b_sgs_slip) {
            cti::utils::initAndCopyToDevice(sgs_cdel_w_d,sgs_cdel_w,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(sgs_f_w_d,sgs_f_w,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(sij_w_d,(double*)sij_w,6*zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(betaij_w_d,(double*)betaij_w,6*zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(BS_d,BS,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(BB_d,BB,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(BR_d,BR,zone_ptr->nbf);
          }

          if (b_lam_sensor) {
            cti::utils::initAndCopyToDevice(u_avg_d,(double*)u_avg,3*zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(uu_avg_d,uu_avg,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(utau_avg_d,utau_avg,zone_ptr->nbf);
          }

          cti::utils::initOnDevice(res_d,1024);

#endif

	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  
          assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  
          // add the flux...
	  
          const double gogm1 = gamma/(gamma-1.0);

          const double t_c = min(t_avg,time-time_init);
          const double eps_t = dt/(dt+t_c);

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {

              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

              // compute unit_n
              double unit_n[3];
              const double mag_n = MAG(zone_ptr->n_bf[ibf]);
              FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i]/mag_n;

              // update the slip length ...

              if (b_param1) {
                // channel parameterization
                double tau0 = 0.0;
                double tau_n = 0.0;
                FOR_I3 tau_n += tau_wall_[ibf][i] * unit_n[i];
                FOR_I3 tau0 += (tau_wall_[ibf][i]-tau_n*unit_n[i])*(tau_wall_[ibf][i]-tau_n*unit_n[i]);
                if (tau0 > 0.0)
                  tau0 = sqrt(tau0);
                else tau0 = 0.0;
                double yp = sqrt(tau0/rho[icv]) * rho[icv] * delta / mu_lam[icv];

                cdel_w[ibf] = (0.19-0.02*log(yp/13.0+1e-6))*(1.0-exp(-(yp*yp/13.0/13.0))) * delta;

              } else if (b_param2) {

                // channel parameterization
                double tau0 = 0.0;
                double tau_n = 0.0;
                FOR_I3 tau_n += tau_wall_[ibf][i] * unit_n[i];
                FOR_I3 tau0 += (tau_wall_[ibf][i]-tau_n*unit_n[i])*(tau_wall_[ibf][i]-tau_n*unit_n[i]);
                if (tau0 > 0.0)
                  tau0 = sqrt(tau0);
                else tau0 = 0.0;
                double yp = sqrt(tau0/rho[icv]) * rho[icv] * delta / mu_lam[icv];

                cdel_w[ibf] = (0.19-0.02*log(yp/13.0+1e-6))*(1.0-exp(-(yp*yp/13.0/13.0))) * delta;

                // sensor
                double dpdn = 0.0;
                double dpds[3];
                FOR_I3 dpdn += dpdx[icv][i] * unit_n[i];
                FOR_I3 dpds[i] = dpdx[icv][i] - dpdn * unit_n[i];
                double u_p = pow(mu_lam[icv]/rho[icv]/rho[icv]*MAG(dpds),(1.0/3.0));
                double u_tau = sqrt(tau0/rho[icv]);

                if (u_p > u_tau) {
                  cdel_w[ibf] = c_slip * delta;

                  double Ap = 17.0;
                  double damp = 1.0 - exp(-yp/Ap);
                  if (b_nodamp) {
                    damp = 1.0;
                  }
                  cdel_w[ibf] *= damp;
                }

                // laminar sensor (Bodart & Larsson)
                // s_tl = <sqrt(u'_i u'_i/2)>/<u_tau>
                // d<f>/dt = (f-<f>)/T
                // T = 1/sqrt(SijSij)
                // s_tl^0 = 1.4 (from Bodart & Larsson) (s_tl <= 1.4 is laminar)
                if (b_lam_sensor) {
                  double stl_local = uu_avg[ibf] / (utau_avg[ibf] + 1e-12);
                  const double stl0 = 1.4;
                  //const double stl0 = 0.1;
                  const double c_slip_lam = 0.5;
                  if ( stl_local <= stl0 ) {
                    cdel_w[ibf] = c_slip_lam * delta; 
                  } 
                  
                  double Ap = 17.0;
                  double damp = 1.0 - exp(-yp/Ap);
                  if (b_nodamp) {
                    damp = 1.0;
                  }
                  cdel_w[ibf] *= damp;
                }

              } else if (b_param3) {
                
                // channel parameterization
                double tau0 = 0.0;
                double tau_n = 0.0;
                FOR_I3 tau_n += tau_wall_[ibf][i] * unit_n[i];
                FOR_I3 tau0 += (tau_wall_[ibf][i]-tau_n*unit_n[i])*(tau_wall_[ibf][i]-tau_n*unit_n[i]);
                if (tau0 > 0.0)
                  tau0 = sqrt(tau0);
                else tau0 = 0.0;
                double yp = sqrt(tau0/rho[icv]) * rho[icv] * delta / mu_lam[icv];

                cdel_w[ibf] = (0.19-0.02*log(yp/13.0+1e-6))*(1.0-exp(-(yp*yp/13.0/13.0))) * delta;

                // sensor
                double dpdn = 0.0;
                double dpds[3];
                FOR_I3 dpdn += dpdx[icv][i] * unit_n[i];
                FOR_I3 dpds[i] = dpdx[icv][i] - dpdn * unit_n[i];
                double u_p = pow(mu_lam[icv]/rho[icv]/rho[icv]*MAG(dpds),(1.0/3.0));
                double u_tau = sqrt(tau0/rho[icv]);

                double up_dot_utau = 0.0;
                FOR_I3 up_dot_utau += dpds[i] * (tau_wall_[ibf][i]-tau_n*unit_n[i]);

                // if (u_p*pow(yp,(1.0/3.0)) > u_tau) {
                //   cdel_w[ibf] = c_slip * delta;
                // 
                //   double Ap = 17.0;
                //   double damp = 1.0 - exp(-yp/Ap);
                //   if (b_nodamp) {
                //     damp = 1.0;
                //   }
                //   cdel_w[ibf] *= damp;
                // }

                if (u_p > u_tau && up_dot_utau > 0.0) { // adverse pressure gradient
                  cdel_w[ibf] = c_slip * delta;

                  double Ap = 17.0;
                  double damp = 1.0 - exp(-yp/Ap);
                  if (b_nodamp) {
                    damp = 1.0;
                  }
                  cdel_w[ibf] *= damp;
                } else if (u_p > u_tau && up_dot_utau < 0.0) { // favorable pressure gradient
                  cdel_w[ibf] = 2.0*c_slip * delta;

                  double Ap = 17.0;
                  double damp = 1.0 - exp(-yp/Ap);
                  if (b_nodamp) {
                    damp = 1.0;
                  }
                  cdel_w[ibf] *= damp;
                }

              } else {

                cdel_w[ibf]        = c_slip * delta; // nominal value
              
                double magu        = 0.0;
                FOR_I3 magu += (u[icv][i] - u_wall[ibf][i])*(u[icv][i] - u_wall[ibf][i]);

                magu        = sqrt(magu);
                double tau0 = AlgebraicWM::solve_tau(magu,delta,rho[icv],mu_lam[icv]);
                double yp   = sqrt(tau0/rho[icv]) * rho[icv] * delta / mu_lam[icv];
                double Ap   = 17.0;
                double damp = 1.0 - exp(-yp/Ap);

                if (b_nodamp) {
                  damp = 1.0;
                }

                cdel_w[ibf] *= damp;

              }

              // recompute the slip state ... 

	      T_s[ibf] = (T[icv]*cdel_w[ibf] + T_bc*delta)/(cdel_w[ibf]+delta);
              double rho_bc = p[icv] / (R_gas * T_bc);  // actuall rho at the wall, not slip
              if (b_adiabatic) { //XXX
                T_s[ibf] = T[icv];
                rho_bc = p[icv] / (R_gas * T_s[ibf]);
              }
	      FOR_I3 {
		rhou_s[ibf][i] = (rho[icv]*u[icv][i]*cdel_w[ibf] + rho_bc*u_wall[icv][i]*delta)/(cdel_w[ibf]+delta);
              }
              
              // compute the mean transpiration...
              double rhoun_s = 0.0;
              FOR_I3 rhoun_s += rhou_s[ibf][i] * unit_n[i];

              if (t_avg < 0.0) {
                mean_rhoun_s[ibf] = rhoun_s; // we will reduce from here
              } else {
                mean_rhoun_s[ibf] *= (1.0 - eps_t);
                mean_rhoun_s[ibf] += eps_t * rhoun_s;
              }

            }

          }

          if (t_avg < 0.0) {
            // reduce the mean transpiration over all wall faces in this bf zone
            double my_buf[2] = {0.0,0.0};
            double buf[2] = {0.0,0.0};
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              my_buf[0] += zone_ptr->area_bf[ibf];
              my_buf[1] += mean_rhoun_s[ibf] * zone_ptr->area_bf[ibf];
            }
            MPI_Allreduce(my_buf,buf,2,MPI_DOUBLE,MPI_SUM,mpi_comm);
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_rhoun_s[ibf] = buf[1]/buf[0];
            }
          }

	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	
            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      
              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
              
              // compute unit_n
              double unit_n[3];
              const double mag_n = MAG(zone_ptr->n_bf[ibf]);
              FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i]/mag_n;

              // correct the slip state ...
	  
              FOR_I3 rhou_s[ibf][i] -= mean_rhoun_s[ibf] * unit_n[i];

              // recompute sgs slip parameters

              double rho_bc = p[icv] / (R_gas * T_bc); // actual rho at the wall, not slip
              if (b_adiabatic) { //XXX
                rho_bc = p[icv] / (R_gas * T_s[ibf]);
              }

              double eps_s = 0.0; // also want to use this for laminar sensor
              double eps_s_long = 0.0; // longer time scale
              if (b_sgs_slip) {
                double dudx_w[3][3];
                FOR_I3 {
                  FOR_J3 {
                    dudx_w[i][j] = (rhou_s[ibf][i]/rho_bc - u[icv][i])/delta*unit_n[j];
                  }
                }
                double dudx_dot_n[3] = {0.0,0.0,0.0};
                FOR_I3 {
                  FOR_J3 {
                    dudx_dot_n[i] += dudx[icv][i][j] * unit_n[j];
                  }
                }
                FOR_I3 {
                  FOR_J3 {
                    dudx_w[i][j] += dudx[icv][i][j] - dudx_dot_n[i]*unit_n[j];
                  }
                }

                for (int i = 0; i < 3; ++i) {
                  sij_w[ibf][i] = dudx_w[i][i];
                  sij_w[ibf][i+3] = 0.5*(dudx_w[i][(i+1)%3] + dudx_w[(i+1)%3][i]);
                }
                
                double sij_cv[6];
                for (int i = 0; i < 3; ++i) {
                  sij_cv[i] = dudx[icv][i][i];
                  sij_cv[i+3] = 0.5*(dudx[icv][i][(i+1)%3] + dudx[icv][(i+1)%3][i]);
                }

                double dsdn_w[6];
                for (int i = 0; i < 6; ++i) {
                  dsdn_w[i] = (sij_cv[i] - sij_w[ibf][i])/delta;
                }

                for (int i = 0; i < 6; ++i) {
                  betaij_w[ibf][i] = sij_w[ibf][i] - cdel_w[ibf]*dsdn_w[i];
                }

                double rij_w[6];
                for (int i = 0; i < 3; ++i) {
                  rij_w[i] = rhou_s[ibf][i]*rhou_s[ibf][i]/rho_bc/rho_bc;
                  rij_w[i+3] = rhou_s[ibf][i]*rhou_s[ibf][(i+1)%3]/rho_bc/rho_bc;
                }
                // // remove the trace of rij
                // double rkk = rij_w[0] + rij_w[1] + rij_w[2];
                // for (int i = 0; i < 3; ++i)
                //   rij_w[i] -= rkk/3.0;

                // compute averaging timescale (1/|S|)
                double mag_S = 0.0;
                FOR_I3 {
                  mag_S += sij_w[ibf][i]*sij_w[ibf][i];
                  mag_S += 2.0*sij_w[ibf][i+3]*sij_w[ibf][i+3];
                }
                mag_S *= 2.0;
                double t_scale = 0.0;
                if (mag_S > 0.0) t_scale = 1.0/sqrt(mag_S);
                //const double eps_s = dt/(dt+t_scale);
                eps_s = dt/(dt+t_scale);
                eps_s_long = dt/(dt+100.0*t_scale);
                
                double BSc = 0.0;
                double BBc = 0.0;
                double BRc = 0.0;
                for (int i = 0; i < 3; ++i) {
                  BBc += betaij_w[ibf][i]*betaij_w[ibf][i];
                  BSc += betaij_w[ibf][i]*sij_w[ibf][i];
                  BRc += betaij_w[ibf][i]*rij_w[i];
                  BBc += 2.0*betaij_w[ibf][3+i]*betaij_w[ibf][3+i];
                  BSc += 2.0*betaij_w[ibf][3+i]*sij_w[ibf][3+i];
                  BRc += 2.0*betaij_w[ibf][3+i]*rij_w[3+i];
                }
                BB[ibf] = (1.0-eps_s)*BB[ibf] + eps_s*BBc;
                BS[ibf] = (1.0-eps_s)*BS[ibf] + eps_s*BSc;
                BR[ibf] = (1.0-eps_s)*BR[ibf] + eps_s*BRc;

                assert(BR[ibf] == BR[ibf]);
                assert(BS[ibf] == BS[ibf]);
                assert(BB[ibf] == BB[ibf]);
                sgs_cdel_w[ibf] = cdel_w[ibf] * BS[ibf] / (BB[ibf] + 1e-12);
                sgs_f_w[ibf] = -0.5 * BR[ibf] / (BB[ibf] + 1e-12);

                assert( sgs_cdel_w[ibf] >= 0.0);
              }

              // now the flux...
	      
              const double rhou_s_dot_n = DOT_PRODUCT(rhou_s[ibf],zone_ptr->n_bf[ibf]);
              const double rho_wall     = p[icv] / (R_gas * T_s[ibf]);
              
              double Frhou[3] = { 0.0, 0.0, 0.0 };
	     
              if (rhou_s_dot_n > 0.0) {
		
                const double H0 = gogm1*p[icv]/rho[icv] + 
		  0.5*DOT_PRODUCT(u[icv],u[icv]);
		
                rhs[icv][0] -= rhou_s_dot_n;
		FOR_I3 {
		  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		  rhs[icv][1+i] -= rhou_s_dot_n*u[icv][i];
                  Frhou[i]      -= rhou_s_dot_n*u[icv][i];

		}
		rhs[icv][4] -= rhou_s_dot_n*H0;
	      }
	      
              else {
	
		const double H_wall = gogm1 * p[icv] / rho_wall + 
		  0.5*DOT_PRODUCT(rhou_s[ibf],rhou_s[ibf])/(rho_wall*rho_wall); 
		
                rhs[icv][0] -= rhou_s_dot_n;
		FOR_I3 {
		  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		  rhs[icv][1+i] -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                  Frhou[i]      -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
		}
		rhs[icv][4] -= rhou_s_dot_n*H_wall;
	      }
	 
              // now the viscous contributions ... 

              double visc_coeff = mu_lam[icv] * area_over_delta_bf[ibf];
              const double l_coeff    = loc_lam[icv]* area_over_delta_bf[ibf];

              // now the sgs contributions ...

              if (b_sgs_slip) {
                const double mu_sgs_w = (delta*sgs_f_w[ibf]+sgs_cdel_w[ibf]*mu_sgs[icv])/(delta+sgs_cdel_w[ibf]);
                visc_coeff += mu_sgs_w * area_over_delta_bf[ibf];

                // nonlinear sgs components (applied using velocity slip BC) // TODO
                // Lij_w - cdel_w (Lij - Lij_w)/dn = 0
                // Lij_w (1 + cdel_w/dn) = cdel_w/dn Lij
                // Lij_w = cdel_w / (dn+cdel_w) Lij
                const double slipfac = cdel_w[ibf]/(delta+cdel_w[ibf]);
                for (int i = 0; i < 3; ++i) {
                  rhs[icv][1+i] += -slipfac * tauijd[icv][i]*n_bf[ibf][i];
                  rhs[icv][1+i] += -slipfac * tauijod[icv][(i+2)%3]*n_bf[ibf][(i+1)%3];
                  rhs[icv][1+i] += -slipfac * tauijod[icv][(i+1)%3]*n_bf[ibf][(i+2)%3];
                  
                  Frhou[i]      += -slipfac * tauijd[icv][i]*n_bf[ibf][i];
                  Frhou[i]      += -slipfac * tauijod[icv][(i+2)%3]*n_bf[ibf][(i+1)%3];
                  Frhou[i]      += -slipfac * tauijod[icv][(i+1)%3]*n_bf[ibf][(i+2)%3];

                  rhs[icv][4]   += -slipfac * tauijd[icv][i]*n_bf[ibf][i]*rhou_s[ibf][i]/rho_wall;
                  rhs[icv][4]   += -slipfac * tauijod[icv][(i+2)%3]*n_bf[ibf][(i+1)%3]*rhou_s[ibf][i]/rho_wall;
                  rhs[icv][4]   += -slipfac * tauijod[icv][(i+1)%3]*n_bf[ibf][(i+2)%3]*rhou_s[ibf][i]/rho_wall;
                }

              }

              for (int i = 0; i < 3; ++i) { 

                double tmp = visc_coeff*(rhou_s[ibf][i]/rho_wall - u[icv][i]);
                rhs[icv][i+1]  += tmp;
                Frhou[i]       += tmp;
                rhs[icv][4]    += tmp * rhou_s[ibf][i] / rho_wall;

              } 

              rhs[icv][4] += l_coeff*(R_gas*gogm1*T_s[ibf] - gogm1*p[icv]/rho[icv]);
           
              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = Frhou[i] / zone_ptr->area_bf[ibf]; // store for later

              // update variables for laminar sensor
              if (b_lam_sensor) {
                FOR_I3 u_avg[ibf][i] = (1.0-eps_s_long)*u_avg[ibf][i] + eps_s_long*u[icv][i];
                double uu_cur = 0.0;
                FOR_I3 uu_cur += (u[icv][i] - u_avg[ibf][i])*(u[icv][i] - u_avg[ibf][i]);
                uu_cur = sqrt( uu_cur / 2.0);
                uu_avg[ibf] = (1.0-eps_s_long)*uu_avg[ibf] + eps_s_long*uu_cur;
                double mag_tau = 0.0;
                FOR_I3 mag_tau += tau_wall_[ibf][i]*tau_wall_[ibf][i];
                mag_tau = sqrt(mag_tau);
                double utau_cur = sqrt( mag_tau / rho[icv]);
                utau_avg[ibf] = (1.0-eps_s_long)*utau_avg[ibf] + eps_s_long*utau_cur;
              }
            
            } else { 

              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = 0.0;

            } 

          }
	}
	
        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {

          delete[] cdel_w;    cdel_w    = NULL;
          delete[] rhou_s;    rhou_s    = NULL;
          delete[] T_s;       T_s       = NULL;
          delete[] u_wall;    u_wall    = NULL;
          delete[] tau_wall_; tau_wall_ = NULL;
          delete[] mean_rhoun_s; mean_rhoun_s = NULL;

          if (b_sgs_slip) {
            delete[] sgs_cdel_w; sgs_cdel_w = NULL;
            delete[] sgs_f_w;    sgs_f_w    = NULL;
            delete[] sij_w;      sij_w      = NULL;
            delete[] betaij_w;   betaij_w   = NULL;
            delete[] BB;         BB         = NULL;
            delete[] BS;         BS         = NULL;
            delete[] BR;         BR         = NULL;
          }

          delete[] u_avg;     u_avg     = NULL;
          delete[] uu_avg;    uu_avg    = NULL;
          delete[] utau_avg;  utau_avg  = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice(cdel_w_d); cdel_w_d = NULL;
          cti::utils::freeOnDevice(rhou_s_d); rhou_s_d = NULL;
          cti::utils::freeOnDevice(T_s_d);    T_s_d    = NULL;
          cti::utils::freeOnDevice(u_wall_d); u_wall_d = NULL;

          cti::utils::freeOnDevice(mean_rhoun_s_d); mean_rhoun_s_d = NULL;

          if (b_sgs_slip) {
            cti::utils::freeOnDevice(sgs_cdel_w_d); sgs_cdel_w_d = NULL;
            cti::utils::freeOnDevice(sgs_f_w_d);    sgs_f_w_d    = NULL;
            cti::utils::freeOnDevice(sij_w_d);      sij_w_d      = NULL;
            cti::utils::freeOnDevice(betaij_w_d);   betaij_w_d   = NULL;
            cti::utils::freeOnDevice(BB);           BB           = NULL;
            cti::utils::freeOnDevice(BS);           BS           = NULL;
            cti::utils::freeOnDevice(BR);           BR           = NULL;
          }

          if (b_lam_sensor) {
            cti::utils::freeOnDevice(u_avg_d);    u_avg_d    = NULL;
            cti::utils::freeOnDevice(uu_avg_d);   uu_avg_d   = NULL;
            cti::utils::freeOnDevice(utau_avg_d); utau_avg_d = NULL;
          }

          cti::utils::freeOnDevice(res_d); res_d = NULL;
#endif

        }
	
        void postLoadBalance() {
          initData();
        }

        void addBoundaryFluxGpu(cti_ffp * rhs, const int rk_stage, gpuStream_t& stream) {
         
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          const double t_c = min(abs(t_avg),time-time_init);
          const double eps_t = dt/(dt+t_c);

          if ( zone_ptr->nbf > 0 ) { 

            {

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/block_size;

              LAUNCH_GPU_KERNEL_STREAM( computeSlipStateAndTransp, grid_size, block_size,
                                        stream, sp_vol_f_d, u_f_d, p_f_d, h_f_d, mu_lam_d,
                                        loc_lam_d,
                                        (cti_ffp)T_bc, u_wall_d, 
                                        cdel_w_d,rhou_s_d,T_s_d,
                                        mean_rhoun_s_d,
                                        tau_wall_d_,
                                        dpdx_d,
                                        c_smag_d,
                                        uu_avg_d,
                                        utau_avg_d,
                                        zone_ptr->cvobf_d, 
                                        zone_ptr->n_bf_d,
                                        zone_ptr->area_bf_d,
                                        zone_ptr->area_over_delta_bf_d, 
                                        zone_ptr->nbf,
                                        (cti_ffp)gamma,
                                        (cti_ffp)eps_t,
                                        (cti_ffp)t_avg,
                                        (cti_ffp)c_slip,
                                        (cti_ffp)R_gas,
                                        b_nodamp,
                                        b_param1,
                                        b_param2,
                                        b_param3,
                                        b_lam_sensor,
                                        b_adiabatic); 

            }

          }
            
          if (t_avg < 0.0) {

            {

              const int block_size = 128;
              const int grid_size = 128;

              LAUNCH_GPU_KERNEL( (computeSlipMeanTransp<cti_ffp,block_size>),
                                  grid_size,block_size,
                                  res_d,
                                  mean_rhoun_s_d,
                                  zone_ptr->area_bf_d,
                                  zone_ptr->nbf);

            }

            {

              // reduce the following in place...

              const int block_size = 128; // must match the grid size above...
              const int grid_size = 1;

              cti::utils::reduce(block_size,block_size,grid_size,res_d,res_d);

            }

            double my_buf, buf;
            cti::utils::copyToHost(&my_buf,res_d,1);

            MPI_Allreduce(&my_buf,&buf,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

            buf /= zone_ptr->area_global; // area average the mean transpiration

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_rhoun_s[ibf] = buf;
            }
            
            cti::utils::copyToDevice( mean_rhoun_s_d,mean_rhoun_s,zone_ptr->nbf);

          }

          if ( zone_ptr->nbf > 0 ) {

            {

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/block_size;

              LAUNCH_GPU_KERNEL_STREAM( addWmSlipIsoFlux, grid_size, block_size, 
                                        stream, rhs,
                                        sp_vol_f_d, u_f_d, p_f_d, h_f_d, mu_lam_d,
                                        loc_lam_d,
                                        dudx_d,
                                        (cti_ffp)T_bc, u_wall_d, 
                                        cdel_w_d,rhou_s_d,T_s_d,
                                        mu_sgs_d,
                                        tauijd_d,
                                        tauijod_d,
                                        sgs_cdel_w_d,
                                        sgs_f_w_d,
                                        sij_w_d,
                                        betaij_w_d,
                                        BB_d,
                                        BS_d,
                                        BR_d,
                                        mean_rhoun_s_d,
                                        u_avg_d,
                                        uu_avg_d,
                                        utau_avg_d,
                                        tau_wall_d_,
                                        c_smag_d,
                                        zone_ptr->cvobf_d, 
                                        zone_ptr->n_bf_d,
                                        zone_ptr->area_bf_d,
                                        zone_ptr->area_over_delta_bf_d, 
                                        zone_ptr->nbf,
                                        (cti_ffp)gamma,
                                        (cti_ffp)eps_t,
                                        (cti_ffp)c_slip,
                                        (cti_ffp)R_gas,
                                        (cti_ffp)dt,
                                        b_sgs_slip,
                                        b_lam_sensor,
                                        b_adiabatic) ; 

            }

          }
#endif
        }
     
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost((double*)rhou_s,rhou_s_d, 3*zone_ptr->nbf);
          cti::utils::copyToHost(cdel_w, cdel_w_d, zone_ptr->nbf);
          cti::utils::copyToHost(T_s,T_s_d, zone_ptr->nbf);

          if (b_sgs_slip) {
            cti::utils::copyToHost(sgs_cdel_w, sgs_cdel_w_d, zone_ptr->nbf);
            cti::utils::copyToHost(sgs_f_w, sgs_f_w_d, zone_ptr->nbf);
            cti::utils::copyToHost((double*)sij_w, sij_w_d, 6*zone_ptr->nbf);
            cti::utils::copyToHost((double*)betaij_w, betaij_w_d, 6*zone_ptr->nbf);
            cti::utils::copyToHost(BB, BB_d, zone_ptr->nbf);
            cti::utils::copyToHost(BS, BS_d, zone_ptr->nbf);
            cti::utils::copyToHost(BR, BR_d, zone_ptr->nbf);
          }

          if (b_lam_sensor) {
            cti::utils::copyToHost((double*)u_avg, u_avg_d, 3*zone_ptr->nbf);
            cti::utils::copyToHost(uu_avg, uu_avg_d, zone_ptr->nbf);
            cti::utils::copyToHost(utau_avg, utau_avg_d, zone_ptr->nbf);
          }

          cti::utils::copyToHost(mean_rhoun_s,mean_rhoun_s_d, zone_ptr->nbf);

          cti::utils::copyToHost((double*)tau_wall_,tau_wall_d_, 3*zone_ptr->nbf);

          // // compute tau_wall for later ... 

          // for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

          //     if (zone_ptr->area_bf[ibf] > 0.0) {
          //     
          //       const int icv          = zone_ptr->cvobf[ibf];
          //       const double p_wall    = p[icv];
          //       const double rho_wall  = p_wall / (R_gas * T_s[ibf]);
          //       const double rhoun_s   = DOT_PRODUCT( rhou_s[ibf], zone_ptr->n_bf[ibf]);

          //       double Frhou[3];

          //       if ( rhoun_s > 0.0) { 

          //         for (int i = 0; i < 3; ++i) 
          //           Frhou[i] = rhoun_s * u[icv][i];
          //       
          //       } else { 

          //         for (int i = 0; i < 3; ++i)
          //           Frhou[i] = rhoun_s*rhou_s[ibf][i]/rho_wall;

          //       } 

          //       const double visc_coeff = mu_lam[icv] * zone_ptr->area_over_delta_bf[ibf];

          //       for (int i = 0; i < 3; ++i) 
          //         Frhou[i] -= visc_coeff*(rhou_s[ibf][i]/rho_wall - u[icv][i]);

          //       for (int i = 0; i < 3; ++i) 
          //         tau_wall_[ibf][i] = Frhou[i] / zone_ptr->area_bf[ibf];
          //     
          //     }
          //     else {
          //      
          //       for (int i = 0; i < 3; ++i) 
          //         tau_wall_[ibf][i] = 0.0;

          //     }
          //   
          //   }
#endif

          if (b_blde_params) {
            blde->computeBlFromPt(bl_delta,bl_delta_star,bl_theta,bl_u_edge,bl_thresh);
          }

        }

        void query() {} 

        void force(double (*rval_dn3)[3], const ForceType f_type) {

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            for (int i =0; i < 3; ++i) 
              rval_dn3[ibf][i] = 0.0;
 
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ){

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ){

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 rval_dn3[ibf][i] += tau_wall_[ibf][i]*zone_ptr->area_bf[ibf]; 
            }
          }
        }

      };

      class WmSlipIsothermalHybridBc : public IdealGasBc {
      public:
	double *cdel_w;
	double (*rhou_s)[3];
        double (*u_wall)[3];
	double *T_s;
	double T_bc;
	double c_slip;

        double (*tau_wall_)[3];

        cti_ffp * cdel_w_d;
        cti_ffp * rhou_s_d; 
        cti_ffp * T_s_d;
        cti_ffp * u_wall_d;

        double (*u1)[3];
        cti_ffp *u1_d;

        WmSlipIsothermalHybridBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  
          if (mpi_rank == 0) 
            cout << "WmSlipIsothermalHybridBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
          zone_ptr->lb_cost = 50;
	  cdel_w = NULL;
	  rhou_s = NULL;
	  
          T_s    = NULL;
          u_wall = NULL;

          cdel_w_d = NULL;
          rhou_s_d = NULL;
          T_s_d    = NULL;
          u_wall_d = NULL;

          u1   = NULL; registerData(u1,zone_ptr->nbf,zone_ptr->name+"u1", READ_DATA|WRITE_DATA);
          u1_d = NULL;

          tau_wall_ = NULL;
	  
	  registerFunc(&WmSlipIsothermalHybridBc::y_plus,this,zone_ptr->name+":y_plus");
	  registerFunc(&WmSlipIsothermalHybridBc::c_delw,this,zone_ptr->name+":c_delw");
	  
          bool b_T_bc = false;

          c_slip      = 0.41;
	  
          int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ((token == "T")||(token == "T_WALL")||(token == "T_BC")) {
	      T_bc = param->getDouble(iarg++);
	      b_T_bc = true;
	    }
	    else if (token == "C_SLIP") {
	      c_slip = param->getDouble(iarg++);
	    }
	    else {
	      CERR("unrecognized WM_SLIP token: " << token);
	    }
	  }
	  //int ierr = 0;
	  if (!b_T_bc) {
	    CERR("WM_SLIP requires params: T <double> C_SLIP [<double> = 0.41]");
	  }
	}
	
        ~WmSlipIsothermalHybridBc() {
	  
          DELETE(cdel_w);
	  DELETE(rhou_s);
	  DELETE(T_s);
          DELETE(u_wall);
          DELETE(u1);
          DELETE(tau_wall_);
	
        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {

          // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

              const int icv       = zone_ptr->cvobf[ibf];
              const double u1_mag = MAG(u1[ibf]);
              const double y1     = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

              const double tau_tmp = AlgebraicWM::solve_tau(u1_mag,y1,rho[icv],mu_lam[icv]);


              rval_dn[ibf] = sqrt(tau_tmp*rho[icv])*y1/mu_lam[icv]; // give it the average-ish value... 
            }
	  }
	  return rval;
	}
	
        CtiData *c_delw(vector<CtiData*>& argVec,const bool b_eval) {

	  if (!argVec.empty()) {
	    eval_cerr << "c_delw() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

              const double y1     = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
              rval_dn[ibf]        = cdel_w[ibf]/y1;
            }
	  }
	  return rval;
	}
	
        CtiData *tau_wall(vector<CtiData*>& argVec,const bool b_eval) {
	  
          if (!argVec.empty()) {
	    eval_cerr << "tau_wall() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  
          if (b_eval) { 
	  
            double *rval_dn = rval->dn();
	    
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

              rval_dn[ibf] = MAG(tau_wall_[ibf]); 
              
            }
            
          }
	
          return rval;
	}
	
        void initData() {

	  cdel_w = new double[zone_ptr->nbf];
	  rhou_s = new double[zone_ptr->nbf][3];
	  T_s    = new double[zone_ptr->nbf];
	 
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
	      cdel_w[ibf] = c_slip*delta;
	    }
	    else {
	      cdel_w[ibf] = 0.0;
	    }
            //FOR_I3 u_bc[ibf][i] = 0.0;
	  }

          assert( tau_wall_ == NULL);
          tau_wall_ = new double[zone_ptr->nbf][3];

          // place holder for parsing STATIONARY, etc for the velocity ..XXX

          u_wall = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            for (int i = 0; i < 3; ++i) 
              u_wall[ibf][i] = 0.0;

          if ( !checkInit(u1,zone_ptr->nbf)) { 

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 
              const int icv = zone_ptr->cvobf[ibf];
              for (int i = 0; i < 3; ++i) 
                u1[ibf][i] = u[icv][i];
            }
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( rhou_s_d   == NULL);
          assert( T_s_d      == NULL);
          assert( cdel_w_d   == NULL);
          assert( u_wall_d   == NULL);
          assert( u1_d       == NULL);

          cti::utils::initOnDevice(rhou_s_d,3*zone_ptr->nbf);
          cti::utils::initOnDevice(T_s_d   ,  zone_ptr->nbf);

          // set an initial guess for the cdel_w .. will be updated
          cti::utils::initAndCopyToDevice(cdel_w_d,cdel_w,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_wall_d,(double*)u_wall,3*zone_ptr->nbf);


          cti::utils::initAndCopyToDevice(u1_d,(double*)u1,3*zone_ptr->nbf);
          
#endif

	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  
          assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  
          // add the flux...
	  
          const double gogm1 = gamma/(gamma-1.0);

	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	
            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      
              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

              if ( cdel_w[ibf] > 0.0 ) { 

                //double magu        = 0.0;
                //for (int i = 0; i < 3; ++i) { 
                //  magu += (u[icv][i] - u_wall[ibf][i])*(u[icv][i] - u_wall[ibf][i]);
                //}

                // recompute the slip state ... 

                T_s[ibf] = (T[icv]*cdel_w[ibf] + T_bc*delta)/(cdel_w[ibf]+delta);
                const double rho_bc = p[icv] / (R_gas * T_bc);  // actuall rho at the wall, not slip
                for (int i = 0; i < 3; ++i)  {
                  rhou_s[ibf][i] = (rho[icv]*u[icv][i]*cdel_w[ibf] + rho_bc*u_wall[icv][i]*delta)/(cdel_w[ibf]+delta);
                }
	      
                // now the flux...

                const double rhou_s_dot_n = DOT_PRODUCT(rhou_s[ibf],zone_ptr->n_bf[ibf]);
                const double rho_wall     = p[icv] / (R_gas * T_s[ibf]);
              
                double Frhou[3] = { 0.0, 0.0, 0.0 };
	     
                if (rhou_s_dot_n > 0.0) {
		
                  const double H0 = gogm1*p[icv]/rho[icv] + 0.5*DOT_PRODUCT(u[icv],u[icv]);
		
                  rhs[icv][0] -= rhou_s_dot_n;
                  for (int i = 0; i < 3; ++i) {
                  
                    rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
                    rhs[icv][1+i] -= rhou_s_dot_n*u[icv][i];
                    Frhou[i]      -= rhou_s_dot_n*u[icv][i];

                  }
                  rhs[icv][4] -= rhou_s_dot_n*H0;
	      
                } else {
	
                  const double H_wall = gogm1 * p[icv] / rho_wall + 
		                        0.5*DOT_PRODUCT(rhou_s[ibf],rhou_s[ibf])/(rho_wall*rho_wall); 
		
                  rhs[icv][0] -= rhou_s_dot_n;
                  for (int i = 0; i < 3; ++i) {
                    rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
                    rhs[icv][1+i] -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                    Frhou[i]      -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                  }
                  rhs[icv][4] -= rhou_s_dot_n*H_wall;
                }
	 
                // now the viscous contributions ... 

                const double visc_coeff = mu_lam[icv] * area_over_delta_bf[ibf];
                //const double l_coeff    = loc_lam[icv]* area_over_delta_bf[ibf];

                for (int i = 0; i < 3; ++i) { 

                  double tmp = visc_coeff*(rhou_s[ibf][i]/rho_wall - u[icv][i]);
                  rhs[icv][i+1]  += tmp;
                  Frhou[i]       += tmp;
                  rhs[icv][4]    += tmp * rhou_s[ibf][i] / rho_wall;

                } 

                // XXX ignore the heat transfer here ... 
                //rhs[icv][4] += l_coeff*(R_gas*gogm1*T_s[ibf] - gogm1*p[icv]/rho[icv]);
           
                for (int i = 0; i < 3; ++i) 
                  tau_wall_[ibf][i] = Frhou[i] / zone_ptr->area_bf[ibf]; // store for later
           
              } else { 

                // close with the standard wall model ... 
                
		const double y1 = delta; 
		
                double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u[icv][i] - u_wall[ibf][i];
		
                const double u_mag   = MAG(u_minus_ubc);
                const double tau_tmp = 	AlgebraicWM::solve_tau(u_mag, y1, rho[icv], mu_lam[icv]);

                for (int i = 0; i < 3; ++i) 
                  tau_wall_[ibf][i] = tau_tmp * u_minus_ubc[i] / u_mag;

                double force[3];
                for (int i = 0; i < 3; ++i) 
                  force[i] = tau_tmp*u_minus_ubc[i]/u_mag*zone_ptr->area_bf[ibf];

                for (int i =0; i < 3; ++i) {
                  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
                  rhs[icv][1+i] -= force[i];
                }
                rhs[icv][4] -= DOT_PRODUCT(force, u_wall[ibf]);
                //rhs[icv][4] -= q_wall[ibf]*zone_ptr->area_bf[ibf];

	      }
	    } else { 

              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = 0.0;

            } 
	  }
        }
	
        void rkStep(const double *rk_wgt,const int rk_stage) {

          if ( rk_stage == 1) { 


            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              // compute a local time scale from the strain rate... 

              const int icv = zone_ptr->cvobf[ibf];

              double smag = 0.0;
              for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) { 
                  const double tmp = 0.5*(dudx[icv][i][j] + dudx[icv][j][i]);
                  smag            += tmp*tmp;
                }
              }
              smag = sqrt(smag);

              const double t_relax = 20.0/smag;
              const double eps     = 1.0 / (t_relax/dt + 1.0);

              for (int i = 0; i < 3; ++i) { 
                u1[ibf][i]        += eps*(u[icv][i] - u1[ibf][i]);
              }

              const double y1      = zone_ptr->area_bf[ibf] / 
                                     zone_ptr->area_over_delta_bf[ibf];


              // we are going to use a yplus trigger here ... 

              const double u1_mag  = MAG(u1[ibf]);
              const double tau_tmp = AlgebraicWM::solve_tau(u1_mag,y1,rho[icv],mu_lam[icv]);

              const double u_tau   = sqrt( tau_tmp / rho[icv]);
              const double yplus   = rho[icv]*u_tau*y1/mu_lam[icv];
              
              if ( yplus > 23.0 ) { 
              
                // the 23 is the buffer layer extension in the alg closure 
                cdel_w[ibf] = 0.0;
              
              } else { 

                cdel_w[ibf] = c_slip * y1 * (1.0-exp(-yplus/17.0));

              } 

            } 
          
          }

        }
	
        void preLoadBalance() {

          delete[] cdel_w;    cdel_w    = NULL;
          delete[] rhou_s;    rhou_s    = NULL;
          delete[] T_s;       T_s       = NULL;
          delete[] u_wall;    u_wall    = NULL;
          delete[] tau_wall_; tau_wall_ = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice(cdel_w_d); cdel_w_d = NULL;
          cti::utils::freeOnDevice(rhou_s_d); rhou_s_d = NULL;
          cti::utils::freeOnDevice(T_s_d);    T_s_d    = NULL;
          cti::utils::freeOnDevice(u_wall_d); u_wall_d = NULL;
          cti::utils::freeOnDevice(u1_d);     u1_d     = NULL;
#endif

        }
	
        void postLoadBalance() {
          initData();
        }

        void addBoundaryFluxGpu(cti_ffp * rhs, const int rk_stage, gpuStream_t& stream) {
         
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1)/block_size;

            LAUNCH_GPU_KERNEL_STREAM( addWmSlipIsoFluxHybrid, grid_size, block_size, 
                                      stream, rhs,
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d, mu_lam_d,
                                      loc_lam_d,
                                      (cti_ffp)T_bc, u_wall_d, 
                                      cdel_w_d,rhou_s_d,T_s_d,
                                      zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d,
                                      zone_ptr->area_bf_d,
                                      zone_ptr->area_over_delta_bf_d, 
                                      zone_ptr->nbf,
                                      (cti_ffp)gamma,
                                      (cti_ffp)c_slip,
                                      (cti_ffp)R_gas) ; 


          } 
#endif
        }
     
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL( updateVelAndSlipLength, 
                               grid_size, block_size, 
                               u1_d, cdel_w_d, u_f_d, dudx_d,
                               sp_vol_f_d,mu_lam_d,
                               zone_ptr->cvobf_d,
                               zone_ptr->area_bf_d,
                               zone_ptr->area_over_delta_bf_d,
                               zone_ptr->nbf,
                               (cti_ffp)c_slip,
                               dt);

          } 

#endif


        }
       

        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost((double*)rhou_s,rhou_s_d, 3*zone_ptr->nbf);
          cti::utils::copyToHost(cdel_w, cdel_w_d, zone_ptr->nbf);
          cti::utils::copyToHost(T_s,T_s_d, zone_ptr->nbf);
          cti::utils::copyToHost((double*)u1, u1_d, 3*zone_ptr->nbf);

          // compute tau_wall for later ... 

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      
              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

              if ( cdel_w[ibf] > 0.0 ) { 

                //double magu        = 0.0;
                //for (int i = 0; i < 3; ++i) { 
                //  magu += (u[icv][i] - u_wall[ibf][i])*(u[icv][i] - u_wall[ibf][i]);
                //}

                //const double rho_bc = p[icv] / (R_gas * T_bc);  // actuall rho at the wall, not slip
	      
                // now the flux...

                const double rhou_s_dot_n = DOT_PRODUCT(rhou_s[ibf],zone_ptr->n_bf[ibf]);
                const double rho_wall     = p[icv] / (R_gas * T_s[ibf]);
              
                double Frhou[3] = { 0.0, 0.0, 0.0 };
	     
                if (rhou_s_dot_n > 0.0) {
		
                  for (int i = 0; i < 3; ++i) 
                    Frhou[i]      -= rhou_s_dot_n*u[icv][i];
                  
                } else {
	
                  for (int i = 0; i < 3; ++i) 
                    Frhou[i]      -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                }
	 
                // now the viscous contributions ... 

                const double visc_coeff = mu_lam[icv] * area_over_delta_bf[ibf];
                //const double l_coeff    = loc_lam[icv]* area_over_delta_bf[ibf];

                for (int i = 0; i < 3; ++i) { 

                  double tmp = visc_coeff*(rhou_s[ibf][i]/rho_wall - u[icv][i]);
                  Frhou[i]       += tmp;

                } 

                // XXX ignore the heat transfer here ... 
                //rhs[icv][4] += l_coeff*(R_gas*gogm1*T_s[ibf] - gogm1*p[icv]/rho[icv]);
           
                for (int i = 0; i < 3; ++i) 
                  tau_wall_[ibf][i] = -Frhou[i] / zone_ptr->area_bf[ibf]; // store for later
           
              } else { 

                // close with the standard wall model ... 
                
		const double y1 = delta; 
		
                double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u[icv][i] - u_wall[ibf][i];
		
                const double u_mag   = MAG(u_minus_ubc);
                const double tau_tmp = 	AlgebraicWM::solve_tau(u_mag, y1, rho[icv], mu_lam[icv]);

                for (int i = 0; i < 3; ++i) 
                  tau_wall_[ibf][i] = tau_tmp * u_minus_ubc[i] / u_mag;

	      }
            } else { 

              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = 0.0;

            } 
          }

#endif
        }

        void query() {} 

        void force(double (*rval_dn3)[3], const ForceType f_type) {

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            for (int i =0; i < 3; ++i) 
              rval_dn3[ibf][i] = 0.0;
 
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ){

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ){

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 rval_dn3[ibf][i] += tau_wall_[ibf][i]*zone_ptr->area_bf[ibf]; 
            }
          }
        }

      };


      class WmSlipIsothermalVBc : public IdealGasBc {
      public:
	double *cdel_w;
	double (*rhou_s)[3];
        double (*u_wall)[3];
	double *T_s;
        //double (*u_bc)[3];
	double T_bc;
	double c_slip;

        double (*tau_wall_)[3];

        cti_ffp * cdel_w_d;
        cti_ffp * rhou_s_d; 
        cti_ffp * T_s_d;
        cti_ffp * u_wall_d;

        bool b_dynamic;
        double *  vsq_filt;
        double *  vsq;
        cti_ffp * vsq_filt_d;
        cti_ffp * vsq_d;


        WmSlipIsothermalVBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  
          if (mpi_rank == 0) 
            cout << "WmSlipIsothermalVBc: " << param->str() << endl;
        
          zone_ptr->b_wall  = true;
          zone_ptr->lb_cost = 50;

          cdel_w    = NULL; 
          registerData(cdel_w,zone_ptr->nbf,zone_ptr->name+":cdel_w", READ_DATA|WRITE_DATA);
	  rhou_s    = NULL;
	  
          T_s       = NULL;
          u_wall    = NULL;

          cdel_w_d  = NULL;
          rhou_s_d  = NULL;
          T_s_d     = NULL;
          u_wall_d  = NULL;
          tau_wall_ = NULL;
	  
          //registerFunc(&WmSlipIsothermalBc::tau_wall,this,zone_ptr->name+":tau_wall");
	  registerFunc(&WmSlipIsothermalVBc::y_plus,this,zone_ptr->name+":y_plus");
	  registerFunc(&WmSlipIsothermalVBc::cslip,this,zone_ptr->name+":cslip");
	  
          bool b_T_bc    = false;
          b_dynamic      = false;
          c_slip         = 0.41;
	  
          int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ((token == "T")||(token == "T_WALL")||(token == "T_BC")) {
	      T_bc = param->getDouble(iarg++);
	      b_T_bc = true;
	    }
	    else if (token == "C_SLIP") {
	      c_slip = param->getDouble(iarg++);
	    }
            else if ( token == "DYNAMIC") { 
              b_dynamic = true;
            }
	    else {
	      CERR("unrecognized WM_SLIP token: " << token);
	    }
	  }
	  //int ierr = 0;
	  if (!b_T_bc) {
	    CERR("WM_SLIP requires params: T <double> C_SLIP [<double> = 0.41]");
	  }


          vsq_filt   = NULL;
          vsq        = NULL;
          vsq_filt_d = NULL;
          vsq_d      = NULL;
          
          if ( b_dynamic ) { 

            registerData(vsq_filt, zone_ptr->nbf, zone_ptr->name+":vsq_filt", READ_DATA | WRITE_DATA);
            registerData(vsq     , zone_ptr->nbf, zone_ptr->name+":vsq", READ_DATA|WRITE_DATA);

          } 

	}
	
        ~WmSlipIsothermalVBc() {
	 
          DELETE(cdel_w);
	  DELETE(rhou_s);
	  DELETE(T_s);
          DELETE(u_wall);
          DELETE(tau_wall_);
          DELETE(vsq_filt);
          DELETE(vsq);

        }
	
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {

          // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		
                // TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		
                // y_plus is then...
		rval_dn[ibf] = y1*sqrt(MAG(tau_wall_[ibf])*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}
	
        CtiData *cslip(vector<CtiData*>& argVec,const bool b_eval) {

	  if (!argVec.empty()) {
	    eval_cerr << "cslip() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }

	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		
                // TODO: this is bad. We should carry area and delta...
		const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
                rval_dn[ibf]       = cdel_w[ibf] / delta;
	      
              }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}
	
        CtiData *tau_wall(vector<CtiData*>& argVec,const bool b_eval) {
	 
          int comp = -1;
          if (argVec.size() == 1) {
            if (argVec[0]->getDatatype() != I_DATA){  
              eval_cerr << "tau_wall() argument expects I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
              return NULL;
            }
            comp = argVec[0]->i();
            if (( comp < 0) || (comp >= 3) ) {
              eval_cerr << "tau_wall() integer argument should be in [0:3), got " << comp << endl;
              return NULL;
            }
          }
          else if (argVec.size() > 1) {
            eval_cerr << "tau_wall() accepts up to 1 argument: got " << argVec.size() << endl;
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) { 
            
            double *rval_dn = rval->dn();
            // default is magnitude
            if (comp == -1) {
              for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
                rval_dn[ibf] = MAG(tau_wall_[ibf]);
              }
            }
            else {
              assert((comp >= 0)&&(comp < 3));
              for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
                rval_dn[ibf] = tau_wall_[ibf][comp];
              }
            }
            //cti::utils::dumpRange(rval_dn,zone_ptr->nbf,"rval_dn");
          }
          return rval;

        }
	
        void initData() {

	  rhou_s = new double[zone_ptr->nbf][3];
	  T_s    = new double[zone_ptr->nbf];
	
          if ( !checkInit(cdel_w,zone_ptr->nbf)) { 
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
                const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
                cdel_w[ibf] = c_slip*delta;
              }
              else {
                cdel_w[ibf] = 0.0;
              }
            }
          }

          assert( tau_wall_ == NULL);
          tau_wall_ = new double[zone_ptr->nbf][3];

          // place holder for parsing STATIONARY, etc for the velocity ..XXX

          u_wall = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            for (int i = 0; i < 3; ++i) 
              u_wall[ibf][i] = 0.0;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( rhou_s_d == NULL);
          assert( T_s_d    == NULL);
          assert( cdel_w_d == NULL);
          assert( u_wall_d == NULL);

          cti::utils::initOnDevice(rhou_s_d,3*zone_ptr->nbf);
          cti::utils::initOnDevice(T_s_d   ,  zone_ptr->nbf);

          // set an initial guess for the cdel_w .. will be updated
          cti::utils::initAndCopyToDevice(cdel_w_d,cdel_w,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_wall_d,(double*)u_wall,3*zone_ptr->nbf);

#endif

          if ( b_dynamic ) { 


            if ( !checkInit(vsq_filt,zone_ptr->nbf) || 
                 !checkInit(vsq     , zone_ptr->nbf)) { 

              if ( mpi_rank == 0 ) 
                cout << zone_ptr->getName() << " : reinit vsq " << endl; 

              for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 
                
                vsq_filt[ibf] = 0.0;
                vsq[ibf]      = 0.0;

              } 

            }

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

            cti::utils::initAndCopyToDevice(vsq_d     , vsq     , zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(vsq_filt_d, vsq_filt, zone_ptr->nbf);

#endif
          } 

	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  
          assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  
          // add the flux...
	  
          const double gogm1 = gamma/(gamma-1.0);
          assert(0);

	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	
            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
	      
              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
	   
              // recompute the slip state ...
              // this approximation is only going to set a slip velocity that is itself 
              // related to the normal component of the velocity 

              const double magn     = MAG(n_bf[ibf]);
              const double un_cv    = DOT_PRODUCT(u[icv],n_bf[ibf])/magn; // un is pointing inward ... 

              double upar[3]; 
              for (int i = 0; i < 3; ++i) 
                upar[i] = u[icv][i] - un_cv * n_bf[ibf][i] / magn;

	      T_s[ibf] = (T[icv]*cdel_w[ibf] + T_bc*delta)/(cdel_w[ibf]+delta);
              //const double rho_bc = p[icv] / (R_gas * T_bc);  // actuall rho at the wall, not slip
	      FOR_I3 {
		rhou_s[ibf][i] = (rho[icv]*un_cv*n_bf[ibf][i]/magn*cdel_w[ibf])/(cdel_w[ibf]+delta); // XXX assume u_wall = 0
	      }
	      
              // now the flux...
	      
              const double rhou_s_dot_n = DOT_PRODUCT(rhou_s[ibf],zone_ptr->n_bf[ibf]);
              const double rho_wall     = p[icv] / (R_gas * T_s[ibf]);
              
              double Frhou[3] = { 0.0, 0.0, 0.0 };
	     
              if (rhou_s_dot_n > 0.0) {
		
                const double H0 = gogm1*p[icv]/rho[icv] + 
		  0.5*DOT_PRODUCT(u[icv],u[icv]);
		
                rhs[icv][0] -= rhou_s_dot_n;
		FOR_I3 {
		  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		  rhs[icv][1+i] -= rhou_s_dot_n*u[icv][i];
                  Frhou[i]      -= rhou_s_dot_n*u[icv][i];

		}
		rhs[icv][4] -= rhou_s_dot_n*H0;
	      }
	      
              else {
	
		const double H_wall = gogm1 * p[icv] / rho_wall + 
		  0.5*DOT_PRODUCT(rhou_s[ibf],rhou_s[ibf])/(rho_wall*rho_wall); 
		
                rhs[icv][0] -= rhou_s_dot_n;
		FOR_I3 {
		  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		  rhs[icv][1+i] -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                  Frhou[i]      -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
		}
		rhs[icv][4] -= rhou_s_dot_n*H_wall;
	      }
	 
              // now the viscous contributions ... which we are evaluating
              // through an algebraic closure on the wall parallel components... 

              const double upar_mag = MAG(upar);
              double tau_par_mag;
              double tau_par[3]; 

              if ( upar_mag > 0.0) { 

                tau_par_mag =  AlgebraicWM::solve_tau(upar_mag,delta,
                                                      rho[icv],mu_lam[icv]);

                //tau_par_mag =  AlgebraicWM_spalding::solve_tau(upar_mag,delta,
                //                                               rho[icv],mu_lam[icv],
                //                                               1.0e-6);


                for (int i = 0; i < 3; ++i) 
                  tau_par[i] = tau_par_mag*upar[i]/upar_mag;

              } else { 

                tau_par_mag = 0.0;
                for (int i = 0; i < 3; ++i) 
                  tau_par[i] = 0.0;

              } 

              
              //const double visc_coeff = mu_lam[icv] * area_over_delta_bf[ibf];
              const double l_coeff    = loc_lam[icv]* area_over_delta_bf[ibf];

              for (int i = 0; i < 3; ++i) { 

                double tmp      = -tau_par[i]*area_bf[ibf]; 
                rhs[icv][i+1]  += tmp;
                Frhou[i]       += tmp;
                rhs[icv][4]    += tmp * rhou_s[ibf][i] / rho_wall; // should eval to 0...

              } 

              rhs[icv][4] += l_coeff*(R_gas*gogm1*T_s[ibf] - gogm1*p[icv]/rho[icv]);
           
              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = tau_par[i]; 
            
            } else { 

              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = 0.0;

            } 

          }
	}
	
        void preLoadBalance() {

          delete[] rhou_s;    rhou_s    = NULL;
          delete[] T_s;       T_s       = NULL;
          delete[] u_wall;    u_wall    = NULL;
          delete[] tau_wall_; tau_wall_ = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice(cdel_w_d); cdel_w_d = NULL;
          cti::utils::freeOnDevice(rhou_s_d); rhou_s_d = NULL;
          cti::utils::freeOnDevice(T_s_d);    T_s_d    = NULL;
          cti::utils::freeOnDevice(u_wall_d); u_wall_d = NULL;

          if ( b_dynamic ) { 

            cti::utils::freeOnDevice(vsq_filt_d); vsq_filt_d = NULL;
            cti::utils::freeOnDevice(vsq_d);      vsq_d      = NULL;

          } 
#endif

        }
	
        void postLoadBalance() {
          initData();
        }

        void rkStep( const double * rk_wgt, const int rk_stage) { 

          if ( rk_stage == 1 ) { 

            if ( !b_dynamic ) { 

              for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

                const int icv      = zone_ptr->cvobf[ibf];
                const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
	
                // update the slip length ... 

                cdel_w[ibf]        = c_slip * delta; // nominal value 

                double magu        = 0.0;
                for (int i = 0; i < 3; ++i) { 
                  magu += (u[icv][i] - u_wall[ibf][i])*(u[icv][i] - u_wall[ibf][i]);
                }

                magu        = sqrt(magu);
                double tau0 = AlgebraicWM::solve_tau(magu,delta,rho[icv],mu_lam[icv]);
                //double tau0 = AlgebraicWM_spalding::solve_tau(magu,delta,rho[icv],mu_lam[icv],1.0e-6);
                double yp   = sqrt(tau0/rho[icv]) * rho[icv] * delta / mu_lam[icv];
                double Ap   = 17.0;
                double damp = 1.0 - exp(-yp/Ap);

                cdel_w[ibf] *= damp;

              } 

            } else { 

              assert(0);

            } 
          } 

        }

        void rkStepGpu( const double * rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( (rk_stage == 1) && (zone_ptr->nbf > 0)) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1)/block_size;

            if ( !b_dynamic ) { 

              LAUNCH_GPU_KERNEL( setConstantSlipLength, grid_size, block_size, 
                                 cdel_w_d, sp_vol_f_d, u_f_d, u_wall_d, mu_lam_d, 
                                 zone_ptr->area_bf_d, zone_ptr->area_over_delta_bf_d,
                                 zone_ptr->cvobf_d,
                                 zone_ptr->nbf, (cti_ffp) c_slip);

            } else { 

              initFilterStructsDevice();
              ensureDudxFiltDevice(u_f_d);

              /*
              LAUNCH_GPU_KERNEL( computeDynamicSlipVLength2, grid_size, block_size, 
                                 cdel_w_d, vsq_filt_d, vsq_d, dudx_filt_d, 
                                 u_f_d, cvocv_i_fa_d, cvocv_v_fa_d, cvocv_filt_d, 
                                 zone_ptr->cvobf_d, zone_ptr->area_bf_d, 
                                 zone_ptr->area_over_delta_bf_d, 
                                 zone_ptr->n_bf_d, zone_ptr->nbf, 
                                 dt);
               */


              LAUNCH_GPU_KERNEL( computeDynamicSlipVLength3, grid_size, block_size, 
                                 cdel_w_d, vsq_filt_d, vsq_d, dudx_d, dudx_filt_d, 
                                 p_f_d, sp_vol_f_d, 
                                 cvocv_i_fa_d, cvocv_v_fa_d, cvocv_filt_d, 
                                 zone_ptr->cvobf_d, zone_ptr->area_bf_d, 
                                 zone_ptr->area_over_delta_bf_d, 
                                 zone_ptr->n_bf_d, zone_ptr->nbf, 
                                 dt);

            } 

          } 

#endif


        }

        void addBoundaryFluxGpu(cti_ffp * rhs, const int rk_stage, gpuStream_t& stream) {
         
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1)/block_size;

            LAUNCH_GPU_KERNEL_STREAM( addWmSlipIsoVFlux, grid_size, block_size, 
                                      stream, rhs,
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d, mu_lam_d,
                                      loc_lam_d,
                                      (cti_ffp)T_bc, u_wall_d, 
                                      cdel_w_d,rhou_s_d,T_s_d,
                                      zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d,
                                      zone_ptr->area_bf_d,
                                      zone_ptr->area_over_delta_bf_d, 
                                      zone_ptr->nbf,
                                      (cti_ffp)gamma,
                                      (cti_ffp)c_slip,
                                      (cti_ffp)R_gas) ; 


          } 
#endif
        }
     
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost((double*)rhou_s,rhou_s_d, 3*zone_ptr->nbf);
          cti::utils::copyToHost(cdel_w, cdel_w_d, zone_ptr->nbf);
          cti::utils::copyToHost(T_s,T_s_d, zone_ptr->nbf);

          if ( b_dynamic ) { 

            cti::utils::copyToHost(vsq_filt,vsq_filt_d, zone_ptr->nbf);
            cti::utils::copyToHost(vsq     ,vsq_d     , zone_ptr->nbf);

          } 

          // compute tau_wall for later ... 

          double *cslip = new double[zone_ptr->nbf];

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const int icv = zone_ptr->cvobf[ibf];

            if (zone_ptr->area_bf[ibf] > 0.0) {
              
              const double magn     = MAG(zone_ptr->n_bf[ibf]);
              const double un_cv    = DOT_PRODUCT(u[icv],zone_ptr->n_bf[ibf])/magn; 
              const double delta    = zone_ptr->area_bf[ibf] / zone_ptr->area_over_delta_bf[ibf];

              double upar[3]; 
              for (int i = 0; i < 3; ++i) 
                upar[i] = u[icv][i] - un_cv * zone_ptr->n_bf[ibf][i] / magn;


              const double upar_mag = MAG(upar);
              const double tau_par  = AlgebraicWM::solve_tau(upar_mag,delta,
                                                             rho[icv],mu_lam[icv]);

              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i]   = tau_par*upar[i]/upar_mag;
	         
              cslip[ibf] = cdel_w[ibf] / delta;
            }
            else {
               
              for (int i = 0; i < 3; ++i) 
                tau_wall_[ibf][i] = 0.0;

              cslip[ibf] = 0.0;
            }

          }

          cti::utils::dumpRange(cslip,zone_ptr->nbf,zone_ptr->getName()+":cslip");
          delete[] cslip;

#endif
        }

        void query() {} 

        void force(double (*rval_dn3)[3], const ForceType f_type) {

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            for (int i =0; i < 3; ++i) 
              rval_dn3[ibf][i] = 0.0;
 
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ){

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        
          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ){

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 rval_dn3[ibf][i] += tau_wall_[ibf][i]*zone_ptr->area_bf[ibf]; 
            }
          }
        }

      };



      class WmiAlgAdiabaticBc : public IdealGasBc {
      public:
	double *tau_wall;
	double *y_interp;
	double (*u_bc)[3]; 
	int *icv_interp;
	double *factor;
	
        WmiAlgAdiabaticBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
	  if (mpi_rank == 0) cout << "WmiAlgAdiabaticBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
	  zone_ptr->lb_cost = 200;
	  tau_wall   = NULL; 
	  y_interp   = NULL; 
	  icv_interp = NULL; 
          u_bc       = NULL;
	  factor     = NULL;
	  registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall",READ_DATA|WRITE_DATA);
	  registerData(y_interp,zone_ptr->nbf,zone_ptr->name+":y_interp"); // no need to store - gets recomputed
	  registerFunc(&WmiAlgAdiabaticBc::y_plus,this,zone_ptr->name+":y_plus");
	}
	
        ~WmiAlgAdiabaticBc() {
	  DELETE(tau_wall);
	  DELETE(y_interp);
	  DELETE(icv_interp);
          DELETE(u_bc);
	  DELETE(factor);
	}
	
        void initData() {
	  assert(tau_wall); // registered data is allocated automatically
	  assert(y_interp); // "
	  assert(icv_interp == NULL); // not registered, so not allocated
	  // set y_interp and icv_interp...
	  // build a csr structure that stores the nearby cvs based on 
	  // extended face connectivity...
	  int *cv_flag = new int[ncv];
	  // 1. start by flagging all cvs associated with bf's for this zone...
	  FOR_ICV cv_flag[icv] = 0;
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    cv_flag[icv] = 1;
	  }
	  // 2. count: include self...
	  int *cvocv_efbf_i = new int[ncv+1];
	  FOR_ICV cvocv_efbf_i[icv+1] = cv_flag[icv]; // add self
	  // 3. include cvs with ef connections...
	  for (int ief = 0; ief < nef; ++ief) {
	    const int icv0 = cvoef[ief][0]; assert((icv0 >= 0)&&(icv0 < ncv));
	    const int icv1 = cvoef[ief][1]; assert((icv1 >= 0)&&(icv1 < ncv_g2));
	    if (cv_flag[icv0] == 1) {
	      cvocv_efbf_i[icv0+1] += 1;
	    }
	    if ((icv1 < ncv)&&(cv_flag[icv1] == 1)) {
	      cvocv_efbf_i[icv1+1] += 1;
	    }
	  }
	  // 4. build csr...
	  cvocv_efbf_i[0] = 0;
	  FOR_ICV cvocv_efbf_i[icv+1] += cvocv_efbf_i[icv];
	  const int cvocv_efbf_s = cvocv_efbf_i[ncv];
	  int *cvocv_efbf_v = new int[cvocv_efbf_s];
	  // 5. set csr...
	  FOR_ICV {
	    if (cv_flag[icv] == 1) {
	      cvocv_efbf_v[cvocv_efbf_i[icv]++] = icv; // self
	    }
	  }
	  for (int ief = 0; ief < nef; ++ief) {
	    const int icv0 = cvoef[ief][0]; assert((icv0 >= 0)&&(icv0 < ncv));
	    const int icv1 = cvoef[ief][1]; assert((icv1 >= 0)&&(icv1 < ncv_g2));
	    if (cv_flag[icv0] == 1) {
	      cvocv_efbf_v[cvocv_efbf_i[icv0]++] = icv1;
	    }
	    if ((icv1 < ncv)&&(cv_flag[icv1] == 1)) {
	      cvocv_efbf_v[cvocv_efbf_i[icv1]++] = icv0;
	    }
	  }
	  // reset cvocv_efbf_i...
	  for (int icv = ncv-1; icv > 0; --icv) {
	    cvocv_efbf_i[icv] = cvocv_efbf_i[icv-1];
	  }
	  cvocv_efbf_i[0] = 0;
	  delete[] cv_flag; cv_flag = NULL;
	  // now loop through the boundary faces and select the furthest point in the 
	  // wall normal direction for sampling the velocity. TODO: we could use an interpolation
	  // or averaging process here in the future to make this more robust...
	  icv_interp = new int[zone_ptr->nbf];
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    assert(cvocv_efbf_i[icv+1]-cvocv_efbf_i[icv] >= 1);
	    double dp_max = 0.0;
	    icv_interp[ibf] = -1;
	    for (int coc = cvocv_efbf_i[icv]; coc != cvocv_efbf_i[icv+1]; ++coc) {
	      const int icv_nbr = cvocv_efbf_v[coc];
	      const double dx[3] = DIFF(zone_ptr->x_bf[ibf],x_cv[icv_nbr]);
	      const double dp = DOT_PRODUCT(dx,zone_ptr->n_bf[ibf]); // recall n_bf is outward normal
	      if ((icv_interp[ibf] == -1)||(dp > dp_max)) {
		icv_interp[ibf] = icv_nbr;
		dp_max = dp;
	      }
	    }
	    assert(icv_interp[ibf] != -1);
	    const double mag_n_bf = MAG(zone_ptr->n_bf[ibf]);
	    assert(mag_n_bf > 0.0);
	    y_interp[ibf] = dp_max/mag_n_bf;
	    // compare to y1:
	    //const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
	    //cout << "XXXXXX: " <<  y_interp[ibf]/y1 << endl;
	  }
	  delete[] cvocv_efbf_i;
	  delete[] cvocv_efbf_v;
	  // finally, if tau_wall is unitialized, then set to the current laminar value...
	  if (!checkInit(tau_wall,zone_ptr->nbf)) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const int icv_i = icv_interp[ibf];
	      const double u1 = MAG(u[icv_i]);
	      // set an initial value for tau_wall, this will constitute the
	      // initial guess for the solution when it is finally constructed.
	      tau_wall[ibf] = mu_lam[icv]*u1/y_interp[ibf];
	    }
	  }
	  assert(u_bc == NULL);
	  u_bc = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            FOR_I3 u_bc[ibf][i] = 0.0;
          }
	  // finally the "factor" that multiplies the wall shear...
	  assert(factor == NULL);
	  factor = new double[zone_ptr->nbf];
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const double y = zone_ptr->x_bf[ibf][1];
	    // factor1, factor12...
	    if (zone_ptr->x_bf[ibf][0] > 0.0) {
	      factor[ibf] = 1.0;
	    }
	    else {
	      // on the negative side of the ramp...
	      const double this_factor = y*(0.085-y)/(0.0425*0.0425); // 0 .. 1
	      factor[ibf] = 1.0 + 2.0*this_factor; // augment stress
	    }
	    // factor2...
	    //const double this_factor = min(y/0.03,1.0); // 0 .. 1
	    //factor[ibf] = 1.0 - 0.95*this_factor; // reduce stress significantly up ramp...
	  }
	}
	CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the wall normal distance in plus units of
	  // the furthest cv from the boundary as stored in icv_interp...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      //const int icv_i = icv_interp[ibf];
	      // y_plus is then...
	      rval_dn[ibf] = y_interp[ibf]*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	    }
	  }
	  return rval;
	}

        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // since the wall model is relatively expensive, we just compute a 
	  // frozen value for the whole time step...
	  if (rk_stage == 1) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const int icv_i = icv_interp[ibf];
	      const double u1 = MAG(u[icv_i]);
	      // use the existing tau_wall to set a guess for the u_tau
	      double this_tau_wall = tau_wall[ibf]/factor[ibf];
	      const double u_tau = sqrt(this_tau_wall / rho[icv]);
	      tau_wall[ibf] = factor[ibf]*AlgebraicWM::solve_tau(u1, y_interp[ibf], rho[icv], mu_lam[icv], u_tau);
	    }
	  }
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    const int icv_i = icv_interp[ibf];
	    // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    const double u1 = MAG(u[icv_i]);
	    if (u1 > 0.0) {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= tau_wall[ibf]*u[icv_i][i]/u1*zone_ptr->area_bf[ibf];
	      }
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	  }
	}
	void rkStep(const double *rk_wgt,const int rk_stage) {}
	void preLoadBalance() {} // TODO: delete[] icv_interp?
	void postLoadBalance() {}

        void addBoundaryFluxGpu(cti_ffp * rhs, const int rk_stage, gpuStream_t& stream) {
          assert(0);
        }
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {}

        void query() {}

        void force(double (*rval_dn3)[3], const ForceType f_type) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          assert(0);
          //force_bf_tauvar_device(rval_dn3,f_type,this); 
#else
          force_bf_tauvar(rval_dn3,f_type,this); 
#endif
        }
      };

      class WmRoughKESAdiabaticBc : public IdealGasBc {
      public:
        
        double * u_wall;
        double * xrot_wall;
        double * axisrot_wall;
        double omega_wall;

        bool b_stationary;
        bool b_use_bfocv;

      public:

        double (*u_bc)[3];
        double *tau_wall;

        double krms;
        double ES;
        double sk;

        cti_ffp * u_bc_d;
        cti_ffp * tau_wall_d;

        string helpString() {
          return
            "\nExamples:"
            "\n  <zonename> WM_ROUGH_KES_ADIABATIC"
            "\n  KRMS <double> ES <double> SKEW <double>"
            "\n  [TRANSLATE <double> <double> <double>]"
            "\n  [ROTATING <double> <double> <double> (xc) <double> <double> <double> (axis) <double> (rpt)]"
            "\n [STATIONARY]";
        }

        WmRoughKESAdiabaticBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {

          if (mpi_rank == 0) cout << "WmRoughKESAdiabaticBc: " << param->str() << endl;
          zone_ptr->b_wall = true;
          zone_ptr->lb_cost = 200;

          u_bc       = NULL;
          tau_wall   = NULL;
          u_bc_d     = NULL;
          tau_wall_d = NULL;

          u_wall       = NULL;
          xrot_wall    = NULL;
          axisrot_wall = NULL;
          omega_wall   = BIG_DOUBLE;
      
	  b_stationary = false;

          // XXX make up some default values
          // model has some issues when k=0, etc.
          krms = 10.0;
          ES = 0.1;
          sk = 0.0;

          registerData(tau_wall,zone_ptr->nbf,zone_ptr->name+":tau_wall", READ_DATA | WRITE_DATA);
	  registerFunc(&WmRoughKESAdiabaticBc::y_plus,this,zone_ptr->name+":y_plus");
	  
          int iarg = 1;
	  while ( iarg < param->size()) {
	    string token = param->getUCString(iarg++);
            if ( token == "KRMS") {
              krms = param->getDouble(iarg++);
              if (mpi_rank == 0) cout << " > KRMS: " << krms << endl; 
            }
            else if ( token == "ES") {
              ES = param->getDouble(iarg++);
              if (mpi_rank == 0) cout << " > ES: " << ES << endl; 
            }
            else if ( token == "SKEW") {
              sk = param->getDouble(iarg++);
              if (mpi_rank == 0) cout << " > SKEW: " << sk << endl; 
            }
	    else if ( token == "TRANSLATE") {
	      assert(u_wall == NULL);
	      u_wall = new double[3];
	      FOR_I3 u_wall[i] = param->getDouble(iarg++);
	      if (mpi_rank == 0) cout << " > Translating wall with velocity: " << COUT_VEC(u_wall) << endl;
	    }
            else if ( token == "ROTATING") {
	      assert(xrot_wall    == NULL);
	      assert(axisrot_wall == NULL);
	      xrot_wall    = new double[3];
	      axisrot_wall = new double[3];
	      FOR_I3 xrot_wall[i]    = param->getDouble(iarg++);
              FOR_I3 axisrot_wall[i] = param->getDouble(iarg++);
              const double mag = sqrt(DOT_PRODUCT(axisrot_wall,axisrot_wall));
              assert ( mag > 0.0) ;
              FOR_I3 axisrot_wall[i] /= mag ;   // make sure it is a unit vector
              omega_wall = param->getDouble(iarg++)*M_PI*2.0 ;   // revolutions per unit time
	      if (mpi_rank == 0)
		cout << " > Rotating wall: center: " << COUT_VEC(xrot_wall)
		     << ", axis: " << COUT_VEC(axisrot_wall) << ", omega: " << omega_wall << endl;
            } 
            else if ( (token == "STATIONARY") || (token == "STATIONARY_FRAME")) {
	      if (frame_rotation) {
		b_stationary = true;
		if (mpi_rank == 0) cout << " > Stationary wall in MRF" << endl;
	      }
	    }
	    else {
	      CERR("unrecognized token " << token << 
		   " during parsing of bc:\n" << param->str() << helpString()); 
	    }
          }
          
          // define the bfocv behavior for this bc application.  it's defaulted
          // as true for hip and false for cuda, but we use the boolean to offer
          // the ability for either to toggle

#ifdef BUILD_CUDA

          b_use_bfocv = getBoolParam("USE_BFOCV", false);

#elif BUILD_HIP

          b_use_bfocv = getBoolParam("USE_BFOCV", false);

#else

          b_use_bfocv = false;  // wont get used at all in cpu land

#endif
        }
        
        ~WmRoughKESAdiabaticBc() {
	  
          DELETE(tau_wall);
	  DELETE(u_bc);
	  DELETE(u_wall);
	  DELETE(xrot_wall);
	  DELETE(axisrot_wall);

        }
        
        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
	  // ------------------------------------------------------------
	  // y_plus() returns the y+ of the bf cell centroid (i.e. effectively 
	  // the cv half-height in viscous units)...
	  // ------------------------------------------------------------
	  if (!argVec.empty()) {
	    eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
	    return NULL;
	  }
	  CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
	  if (b_eval) { 
	    double *rval_dn = rval->dn();
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		// TODO: this is bad. We should carry area and delta...
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		const int icv = zone_ptr->cvobf[ibf];
		// y_plus is then...
		rval_dn[ibf] = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];
	      }
	      else {
		rval_dn[ibf] = 0.0;
	      }
	    }
	  }
	  return rval;
	}
	
        void initData() {

	  // init tau_wall...
	  
          assert(tau_wall);
	  
          if (!checkInit(tau_wall,zone_ptr->nbf)) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      const double u1 = MAG(u[icv]);
	      // set an initial value for tau_wall, this will constitute the
	      // initial guess for the solution when it is finally constructed.
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
		tau_wall[ibf] = mu_lam[icv]*u1/y1;
	      }
	      else {
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  
          // init u_bc...
	  
          assert( u_bc == NULL); u_bc = new double[zone_ptr->nbf][3];
       	  if (u_wall) {
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i  < 3; ++i)
		u_bc[ibf][i] = u_wall[i];
	    }
	  }
	  else if (xrot_wall && axisrot_wall && (omega_wall != BIG_DOUBLE)) {
	    double r[3];
	    for (int ibf =0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - xrot_wall[i];
	      double axis_cross_r[3] = CROSS_PRODUCT(axisrot_wall,r);
	      FOR_I3 u_bc[ibf][i] = omega_wall * axis_cross_r[i];
	    }
	  } 
	  else if (b_stationary) { //TODO consider no "else" here to allow trans/rotate in staionary frame?
	    assert(frame_rotation);
	    double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_bc[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];
	      u_bc[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];
	      u_bc[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }
	  else {
	    // default behavior is u=0 wall
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      for (int i = 0; i < 3; ++i)
		u_bc[ibf][i] = 0.0;
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( u_bc_d == NULL);
	  assert( tau_wall_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(tau_wall_d,tau_wall,zone_ptr->nbf);

          if ( b_use_bfocv) { 

            if (mpi_rank == 0 ) 
              cout << " building bfocv for zone : " << zone_ptr->getName() << endl;
              
            zone_ptr->buildBfocvGpu();

          } 

#endif

	}
        
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...
	  assert((rk_stage >= 1)&&(rk_stage <= NRK));
	  // since the wall model is relatively expensive, we just compute a 
	  // frozen value for the whole time step...
	  if (rk_stage == 1) {
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
		const double y1    = zone_ptr->area_bf[ibf] / zone_ptr->area_over_delta_bf[ibf];
		double u_minus_ubc[3];
		FOR_I3 u_minus_ubc[i] = u[icv][i] - u_bc[ibf][i];
		const double u1 = MAG(u_minus_ubc);
		// use the existing tau_wall to set a guess for the u_tau
		const double u_tau = sqrt(tau_wall[ibf] / rho[icv]);
		tau_wall[ibf] = AlgebraicKESWM::solve_tau(u1, y1, krms, ES, sk, rho[icv], mu_lam[icv], u_tau);
		
	      } 
	      else {
		// assert(zone_ptr->area_bf[ibf] < 1.0E-16); //this should be a collapsed face.
		tau_wall[ibf] = 0.0;
	      }
	    }
	  }
	  // add the flux...
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int icv = zone_ptr->cvobf[ibf];
	    // tau_wall has already been computed at this point, but we'll assume
	    // that the wall stress is instantaneously aligned with the velocity vector
	    //XXXXXXXXXXXXX should we use u or u parallel here??
	    double u_mag = 0.0;
	    for (int i =0; i < 3; ++i)
	      u_mag += (u[icv][i] - u_bc[ibf][i]) * (u[icv][i] - u_bc[ibf][i]);
	    u_mag = sqrt(max(0.0, u_mag));
	    double force[3];
	    if (u_mag > 0.0) {
	      FOR_I3 force[i] = tau_wall[ibf]*(u[icv][i] - u_bc[ibf][i])/u_mag*zone_ptr->area_bf[ibf];
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
		rhs[icv][1+i] -= force[i];
	      }
	      rhs[icv][4] -= DOT_PRODUCT(force, u_bc[ibf]);
	    }
	    else {
	      for (int i =0; i < 3; ++i) {
		rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
	      }
	    }
	  }
	}
        
        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( !b_use_bfocv ) { 

            if ( zone_ptr->nbf > 0 ) { 

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

              LAUNCH_GPU_KERNEL_STREAM( addWmRoughKESAdiabaticFlux, grid_size, 
                                        block_size, stream, rhs, 
                                        sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                        mu_lam_d, u_bc_d, tau_wall_d,
                                        zone_ptr->cvobf_d,
                                        zone_ptr->area_bf_d, 
                                        zone_ptr->area_over_delta_bf_d,
                                        zone_ptr->n_bf_d,
                                        zone_ptr->nbf,
                                        (cti_ffp)krms,
                                        (cti_ffp)ES,
                                        (cti_ffp)sk);

            } 

          }
          else { 

            // avoid the use of atomic adds via a warp like structure 
            // that operates on bfocv.

            assert( zone_ptr->ncv_zone >= 0);

            if ( zone_ptr->ncv_zone > 0 ) { 

              assert( zone_ptr->bfocv_i_d);
              assert( zone_ptr->bfocv_v_d);
              assert( zone_ptr->cv_index_d);
         
              const int coop           = 16;
              const int repeat         = 1;
              const int block_size     = 128;
              const int grid_size      = 1 + (zone_ptr->ncv_zone *coop-1)/(repeat * block_size);

              LAUNCH_GPU_KERNEL_STREAM( (addWmRoughKESFluxCoop<cti_ffp,block_size,coop>),
                                        grid_size, block_size, stream, rhs,
                                        sp_vol_f_d, u_f_d, p_f_d, h_f_d, 
                                        mu_lam_d, u_bc_d, tau_wall_d,
                                        zone_ptr->cv_index_d,
                                        zone_ptr->bfocv_i_d,
                                        zone_ptr->bfocv_v_d,
                                        zone_ptr->area_bf_d, 
                                        zone_ptr->area_over_delta_bf_d,
                                        zone_ptr->n_bf_d,
                                        zone_ptr->ncv_zone,
                                        (cti_ffp)krms,
                                        (cti_ffp)ES,
                                        (cti_ffp)sk, repeat);

            }

          }

#endif

        }
        
        void rkStep(const double *rk_wgt,const int rk_stage) {}
	
        void preLoadBalance() {
          
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
          cti::utils::freeOnDevice(tau_wall_d); tau_wall_d = NULL;

          if ( b_use_bfocv) 
            zone_ptr->destroyBfocvGpu();
#endif
        }
	
        void postLoadBalance() {
          initData();
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::copyToHost(tau_wall,tau_wall_d,zone_ptr->nbf);
#endif

	}

        void query() {

          double my_buf[3] = {0.0,0.0,0.0};
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            const int icv = zone_ptr->cvobf[ibf];

            if ( zone_ptr->area_over_delta_bf[ibf] > 0.0) { 

              const double y1  = zone_ptr->area_bf[ibf] / 
                                 zone_ptr->area_over_delta_bf[ibf];

              const double y_plus = y1*sqrt(tau_wall[ibf]*rho[icv])/mu_lam[icv];

              my_buf[0]          += zone_ptr->area_bf[ibf];
              my_buf[1]          += zone_ptr->area_bf[ibf]*tau_wall[ibf];
              my_buf[2]          += zone_ptr->area_bf[ibf]*y_plus;


            } 
          }

          double buf[3];
          MPI_Reduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0 ) {

            cout << "QUERY_BC " << zone_ptr->getName() 
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:int(tau_wall)dA " << buf[1] 
                 << " 10:y_plus " << buf[2]/buf[0] 
                 << endl; 
          }
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          force_bf_tauvar_device(rval_dn3,f_type,this);
#else 
          force_bf_tauvar(rval_dn3,f_type,this);
#endif
        }

      };
     
      class WmOptSlipBc : public IdealGasBc {
      public:
        double *cdel_w;
        double (*rhou_s)[3];
        double (*u_wall)[3];
        double *T_s;
        double T_bc;
        double c_slip;
        double time_init;

        double * mean_rhoun_s;
        double * mean_direc;
        double (*mean_u1)[3];

        bool b_sgs_slip;
        double *sgs_cdel_w;
        double * sgs_f_w;
        double (*sij_w)[6];
        double (*betaij_w)[6];

        double (*tau_wall_)[3];
        cti_ffp * tau_wall_d_;

        cti_ffp * cdel_w_d;
        cti_ffp * rhou_s_d;
        cti_ffp * T_s_d;
        cti_ffp * u_wall_d;

        cti_ffp * mean_rhoun_s_d;
        cti_ffp * mean_direc_d;
        cti_ffp * mean_u1_d;

        cti_ffp * sgs_cdel_w_d;
        cti_ffp * sgs_f_w_d;
        cti_ffp * sij_w_d;
        cti_ffp * betaij_w_d;

        bool b_filename;
        string filename;
        double *cf_ref;
        cti_ffp * cf_ref_d;

        double t_avg;
        double * GLtn_avg;
        double * DLtn_avg;
        double * Stn_avg;
        double U_inf;

        bool b_modelgrad;
        double * gradJmodel;
        double * hessJmodel;

        cti_ffp * GLtn_avg_d;
        cti_ffp * DLtn_avg_d;
        cti_ffp * Stn_avg_d;

        cti_ffp * gradJmodel_d;
        cti_ffp * hessJmodel_d;

        double dx_fe;
        int nfe_bc;
        double * xfe_bc;
        double * cslip_fe;
        int * feobf;

        int ife_optim;
        int ife_max_error;
        double dt_optim_interval;
        double optim_prev_time;
        double prev_optim_cslip;
        double Lobj;
        double Lobj_prev;
        double cf_area_avg;
        double cf_ref_avg;
        double l2_err;
        double * error_avg;
        double * error_counts;
        double * error_avg_d;
        double * error_counts_d;
        double * cf_avg;
        double * cf_counts;
        double * cf_avg_d;
        double * cf_counts_d;

        const double DC_STEP = 0.01;

        double * gradJmodel_fe;
        double * hessJmodel_fe;

        double * Lobj_eps_net;
        int n_eps_net;
        int idx_eps_net;
        bool b_eps_net;
        double eps_net_max;
        double eps_net_min;
        double cslip_prev_eps_net;

        enum OptimState { EVAL_P1, EVAL_P2, TERMINAL, EPS_NET, GRAD_MODEL };
        OptimState optim_state;
        int max_optim_steps;
        int n_optim_steps;
        double step_optim;

        BoundaryLayerDataExchanger * blde;
        int nbl;
        double l_bl;
        bool b_blde_params;
        const double bl_thresh = 0.99;

        double * bl_delta;
        double * bl_delta_star;
        double * bl_theta;
        double * bl_u_edge;

        double * res_d;

        WmOptSlipBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
        
          if (mpi_rank == 0) cout << "WmOptSlipBc: " << param->str() << endl;
          zone_ptr->lb_cost = 50;
          cdel_w = NULL;
          rhou_s = NULL;

          mean_rhoun_s = NULL;
          mean_direc   = NULL;
          mean_u1      = NULL;

          b_sgs_slip = false;
          sgs_cdel_w = NULL;
          sgs_f_w    = NULL;
          sij_w      = NULL;
          betaij_w   = NULL;

          tau_wall_   = NULL;
          tau_wall_d_ = NULL;

          T_s    = NULL;
          u_wall = NULL;

          cdel_w_d = NULL;
          rhou_s_d = NULL;
          T_s_d    = NULL;
          u_wall_d = NULL;

          mean_rhoun_s_d = NULL;
          mean_direc_d   = NULL;
          mean_u1_d      = NULL;

          sgs_cdel_w_d = NULL;
          sgs_f_w_d    = NULL;
          sij_w_d      = NULL;
          betaij_w_d   = NULL;

          b_filename = false;
          cf_ref     = NULL;
          cf_ref_d   = NULL;

          time_init = 0.0;
          t_avg = 0.0;
          U_inf = BIG_DOUBLE;

          GLtn_avg = NULL;
          DLtn_avg = NULL;
          Stn_avg  = NULL;

          b_modelgrad = false;
          gradJmodel  = NULL;
          hessJmodel  = NULL;
          
          GLtn_avg_d = NULL;
          DLtn_avg_d = NULL;
          Stn_avg_d  = NULL;

          gradJmodel_d = NULL;
          hessJmodel_d = NULL;

          dx_fe = BIG_DOUBLE;

          ife_optim         = BIG_INT;
          ife_max_error     = 0;
          optim_prev_time   = 0;
          dt_optim_interval = BIG_DOUBLE;
          error_avg         = NULL;
          error_counts      = NULL;
          error_avg_d       = NULL;
          error_counts_d    = NULL;
          cf_avg            = NULL;
          cf_counts         = NULL;
          cf_avg_d          = NULL;
          cf_counts_d       = NULL;

          gradJmodel_fe = NULL;
          hessJmodel_fe = NULL;

          optim_state     = EVAL_P1; // initially need to evaluate objective
          max_optim_steps = 10;
          n_optim_steps   = 0;
          step_optim      = 5e-4;

          b_eps_net    = false;
          Lobj_eps_net = NULL;
          n_eps_net    = 0;
          idx_eps_net  = 0;
          eps_net_max  = BIG_DOUBLE;
          eps_net_min  = BIG_DOUBLE;

          blde = NULL;
          nbl = BIG_INT;
          l_bl = BIG_DOUBLE;
          b_blde_params = false;

          bl_delta = NULL;
          bl_delta_star = NULL;
          bl_theta = NULL;
          bl_u_edge = NULL;

          res_d = NULL;

          registerData(mean_rhoun_s,zone_ptr->nbf,zone_ptr->name+":mean_rhoun_s", READ_DATA|WRITE_DATA);

          registerData(cdel_w,zone_ptr->nbf,zone_ptr->name+":cdel_w", READ_DATA|WRITE_DATA);
          registerData(GLtn_avg,zone_ptr->nbf,zone_ptr->name+":GLtn_avg", READ_DATA|WRITE_DATA);
          registerData(DLtn_avg,zone_ptr->nbf,zone_ptr->name+":DLtn_avg", READ_DATA|WRITE_DATA);
          registerData(Stn_avg,zone_ptr->nbf,zone_ptr->name+":Stn_avg", READ_DATA|WRITE_DATA);

          registerData(mean_direc,zone_ptr->nbf,zone_ptr->name+":mean_direc", DEP_DATA);
          registerData(mean_u1,zone_ptr->nbf,zone_ptr->name+":mean_u1", READ_DATA|WRITE_DATA);

          registerData(ife_optim,zone_ptr->name+":ife_optim", READ_DATA|WRITE_DATA);

          registerFunc(&WmOptSlipBc::y_plus,this,zone_ptr->name+":y_plus");
          registerFunc(&WmOptSlipBc::cslip_func,this,zone_ptr->name+":cslip");

          registerFunc(&WmOptSlipBc::grad_cslip,this,zone_ptr->name+":gradient_cslip");
          registerFunc(&WmOptSlipBc::hess_cslip,this,zone_ptr->name+":hessian_cslip");

          registerFunc(&WmOptSlipBc::local_resid,this,zone_ptr->name+":local_resid");
          registerFunc(&WmOptSlipBc::local_error,this,zone_ptr->name+":local_error");
          registerFunc(&WmOptSlipBc::normalized_error,this,zone_ptr->name+":normalized_error");

          registerFunc(&WmOptSlipBc::cf_ref_func,this,zone_ptr->name+":cf_ref");

          bool b_T_bc = false;

          c_slip = 0.41;

          int iarg = 1;
          while (iarg < param->size()) {
            string token = utils::toUpperCase(param->getString(iarg++));
            if ((token == "T")||(token == "T_WALL")||(token == "T_BC")) {
              T_bc = param->getDouble(iarg++);
              b_T_bc = true;
            }
            else if (token == "C_SLIP") {
              c_slip = param->getDouble(iarg++);
            }
            else if (token == "SGS_SLIP") {
              b_sgs_slip = true;
            }
            else if (token == "U_INF") {
              U_inf = param->getDouble(iarg++);
            }
            else if ((token == "DT_AVG")||(token == "T_AVG")||(token == "TIME_AVG")) {
              t_avg = param->getDouble(iarg++);
            }
            else if ((token == "FILE")||(token == "FILENAME")||(token == "FILE_NAME")) {
              filename = param->getString(iarg++);
              COUT(" > FILE " << filename);
              // check that the file is available and readable...
              if (!utils::fileExists(filename)) {
                CERR(" > REFERENCE FILE " << filename << " does not exist");
              }
              b_filename = true;
            }
            else if (token == "DX_FE") {
              dx_fe = param->getDouble(iarg++);
            }
            else if (token == "DT_OPTIM") {
              dt_optim_interval = param->getDouble(iarg++);
            }
            else if (token == "STEP_OPTIM") {
              step_optim = param->getDouble(iarg++);
            }
            else if (token == "MAX_STEPS_OPTIM") {
              max_optim_steps = (int)param->getDouble(iarg++);
            }
            else if (token == "EPS_NET") {
              b_eps_net = true;
              eps_net_min = param->getDouble(iarg++);
              eps_net_max = param->getDouble(iarg++);
              n_eps_net = (int)param->getDouble(iarg++);
            }
            else if (token == "MODEL_GRAD") {
              b_modelgrad = true;
            }
            else {
              CERR(" > unrecognized WM_OPTIMAL_SLIP_ISOTHERMAL token: " << token);
            }
          }
          if (!b_T_bc) {
            CERR(" > WM_OPTIMAL_SLIP_ISOTHERMAL requires params: T <double> C_SLIP [<double> = 0.41] U_INF <double> T_AVG [<double> = 0.0] FILE <string>");
          }

          if (!b_filename) {
            CERR(" > WM_OPTIMAL_SLIP_ISOTHERMAL requires a reference data file");
          }

          if (U_inf == BIG_DOUBLE) {
            CERR(" > WM_OPTIMAL_SLIP_ISOTHERMAL requires a U_inf for Cf calculation");
          }

          if (dx_fe == BIG_DOUBLE) {
            CERR(" > WM_OPTIMAL_SLIP_ISOTHERMAL requires a DX_FE for Cslip discretization");
          }

          if (dt_optim_interval == BIG_DOUBLE) {
            CERR(" > WM_OPTIMAL_SLIP_ISOTHERMAL requires a DT_OPTIM for Cslip optimization");
          }

          Param * blde_param = getParam("BLDE");
          if ( blde_param) {
            iarg = 0;
            while ( iarg < blde_param->size()) {
              string token = blde_param->getString(iarg++);
              if (token == "NBL") {
                nbl = blde_param->getInt(iarg++);
              }
              else if ( token == "L_BL") {
                l_bl = blde_param->getDouble(iarg++);
              }
              else {
                CERR("unrecognized token " << token <<
                     " during parsing of BLDE in bc:\n" << param->str());
              }
            }
          }
          if ( l_bl != BIG_DOUBLE && nbl != BIG_INT) {
            b_blde_params = true;

            registerData(bl_delta,zone_ptr->nbf,zone_ptr->name+":bl_delta",DEP_DATA);
            registerData(bl_delta_star,zone_ptr->nbf,zone_ptr->name+":bl_delta_star",DEP_DATA);
            registerData(bl_theta,zone_ptr->nbf,zone_ptr->name+":bl_theta",DEP_DATA);
            registerData(bl_u_edge,zone_ptr->nbf,zone_ptr->name+":bl_u_e",DEP_DATA);
          }
        
        }
      
        ~WmOptSlipBc() {
        
          DELETE(cdel_w);
          DELETE(rhou_s);
          DELETE(T_s);
          DELETE(u_wall);

          DELETE(mean_rhoun_s);
          DELETE(mean_direc);
          DELETE(mean_u1);

          if (b_sgs_slip) {
            DELETE(sgs_cdel_w);
            DELETE(sgs_f_w);
            DELETE(sij_w);
            DELETE(betaij_w);
          }

          DELETE(GLtn_avg);
          DELETE(DLtn_avg);
          DELETE(Stn_avg);

          if (b_modelgrad) {
            DELETE(gradJmodel);
            DELETE(hessJmodel);

            DELETE(gradJmodel_fe);
            DELETE(hessJmodel_fe);
          }

          DELETE(cslip_fe);
          DELETE(xfe_bc);
          DELETE(feobf);

          DELETE(error_avg);
          DELETE(error_counts);

          DELETE(cf_avg);
          DELETE(cf_counts);

          if (b_eps_net) {
            DELETE(Lobj_eps_net);
          }

          if (blde)
            delete blde;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( cdel_w_d) {
            cti::utils::freeOnDevice(cdel_w_d);
            cdel_w_d = NULL;
          }
          if ( rhou_s_d) {
            cti::utils::freeOnDevice(rhou_s_d);
            rhou_s_d = NULL;
          }
          if ( T_s_d) {
            cti::utils::freeOnDevice(T_s_d);
            T_s_d = NULL;
          }
          if ( u_wall_d) {
            cti::utils::freeOnDevice(u_wall_d);
            u_wall_d = NULL;
          }
          if ( tau_wall_d_) {
            cti::utils::freeOnDevice(tau_wall_d_);
            tau_wall_d_ = NULL;
          }

          if ( mean_rhoun_s_d) {
            cti::utils::freeOnDevice(mean_rhoun_s_d);
            mean_rhoun_s_d = NULL;
          }
          if ( mean_direc_d) {
            cti::utils::freeOnDevice(mean_direc_d);
            mean_direc_d = NULL;
          }
          if ( mean_u1_d) {
            cti::utils::freeOnDevice(mean_u1_d);
            mean_u1_d = NULL;
          }

          if ( sgs_cdel_w_d) {
            cti::utils::freeOnDevice(sgs_cdel_w_d);
            sgs_cdel_w_d = NULL;
          }
          if ( sgs_f_w_d) {
            cti::utils::freeOnDevice(sgs_f_w_d);
            sgs_f_w_d = NULL;
          }
          if ( sij_w_d) {
            cti::utils::freeOnDevice(sij_w_d);
            sij_w_d = NULL;
          }
          if ( betaij_w_d) {
            cti::utils::freeOnDevice(betaij_w_d);
            betaij_w_d = NULL;
          }

          if ( GLtn_avg_d) {
            cti::utils::freeOnDevice(GLtn_avg_d);
            GLtn_avg_d = NULL;
          }
          if ( DLtn_avg_d) {
            cti::utils::freeOnDevice(DLtn_avg_d);
            DLtn_avg_d = NULL;
          }
          if ( Stn_avg_d) {
            cti::utils::freeOnDevice(Stn_avg_d);
            Stn_avg_d = NULL;
          }

          if ( gradJmodel_d) {
            cti::utils::freeOnDevice(gradJmodel_d);
            gradJmodel_d = NULL;
          }
          if ( hessJmodel_d) {
            cti::utils::freeOnDevice(hessJmodel_d);
            hessJmodel_d = NULL;
          }

          if ( error_avg_d) {
            cti::utils::freeOnDevice(error_avg_d);
            error_avg_d = NULL;
          }
          if ( error_counts_d) {
            cti::utils::freeOnDevice(error_counts_d);
            error_counts_d = NULL;
          }
          if ( cf_avg_d) {
            cti::utils::freeOnDevice(cf_avg_d);
            cf_avg_d = NULL;
          }
          if ( cf_counts_d) {
            cti::utils::freeOnDevice(cf_counts_d);
            cf_counts_d = NULL;
          }

          cti::utils::freeOnDevice(res_d); res_d = NULL;

#endif
        
        }

        CtiData *y_plus(vector<CtiData*>& argVec,const bool b_eval) {
        
          // -----------------------------------------------------------
          // y_plus() returns the y+ of the bf cell centroid (i.e. effectively
          // the cv half-height in viscous units)...
          // -----------------------------------------------------------
          if (!argVec.empty()) {
            eval_cerr << "y_plus() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
              
                const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
                const int icv = zone_ptr->cvobf[ibf];

                rval_dn[ibf] = y1*sqrt(MAG(tau_wall_[ibf])*rho[icv])/mu_lam[icv];
              }
              else {
                rval_dn[ibf] = 0.0;
              }
            }
          }
          return rval;
        }
        
        CtiData* cslip_func(vector<CtiData*>& argVec,const bool b_eval) {
        
          if (!argVec.empty()) {
            eval_cerr << "cslip() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
              
                const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

                rval_dn[ibf] = cdel_w[ibf]/y1;
              } else {
                rval_dn[ibf] = 0.0;
              }
            }
          }
          return rval;
        }

        CtiData * grad_cslip(vector<CtiData*>& argVec,const bool b_eval) {
          
          if (!argVec.empty()) {
            eval_cerr << "gradient_cslip() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
           
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {

                const int icv = zone_ptr->cvobf[ibf];
                double delta = zone_ptr->area_bf[ibf] / zone_ptr->area_over_delta_bf[ibf];

                //double cslip_state = cdel_w[ibf]*cdel_w[ibf] / delta / delta;

                double grad_avg = 0.0;

                const double mag_n = MAG(zone_ptr->n_bf[ibf]);
                double unit_n[3];
                FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

                const double tau_wall_n = DOT_PRODUCT(tau_wall_[ibf],unit_n);
                double tau_wall_p[3];
                FOR_I3 tau_wall_p[i] = tau_wall_[ibf][i] - tau_wall_n*unit_n[i];
                tau_wall_p[2] = 0.0;
                double tau_wall_mag = MAG(tau_wall_p);
                if ( tau_wall_p[0] < 0.0) tau_wall_mag *= -1.0;

                { // rhou_s_dot_n <= 0.0

                  double grad = 0.0;
                  grad += 2.0*tau_wall_mag/rho[icv]/U_inf/U_inf;
                  grad += -cf_ref[ibf];
                  grad *= -2.0*delta*delta*GLtn_avg[ibf];

                  grad_avg += grad;
                
                }

                grad_avg *= zone_ptr->area_bf[ibf]/U_inf/U_inf;

                rval_dn[ibf] = grad_avg;

              } else {
                rval_dn[ibf] = 0.0;
              }

            }
          }
          return rval;
        
        }
        CtiData * hess_cslip(vector<CtiData*>& argVec,const bool b_eval) {
          
          if (!argVec.empty()) {
            eval_cerr << "hessian_cslip() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {

                double delta = zone_ptr->area_bf[ibf] / zone_ptr->area_over_delta_bf[ibf];

                //double cslip_state = cdel_w[ibf]*cdel_w[ibf] / delta / delta;

                double inv_step_size_avg = 0.0;

                { // rhou_s_dot_n <= 0.0
                
                  double inv_step_size = 0.0;
                  inv_step_size += 4.0*delta*delta*delta*delta*zone_ptr->area_over_delta_bf[ibf];
                  inv_step_size *= GLtn_avg[ibf]*GLtn_avg[ibf]/U_inf/U_inf/U_inf/U_inf;

                  inv_step_size_avg += inv_step_size;
                
                }

                rval_dn[ibf] = 1.0/inv_step_size_avg;
              
              } else {
                rval_dn[ibf] = 0.0;
              }

            }
          }
          return rval;
        
        }

        CtiData * local_resid(vector<CtiData*>& argVec,const bool b_eval) {
        
          if (!argVec.empty()) {
            eval_cerr << "local_resid() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
              
                const int icv = zone_ptr->cvobf[ibf];

                double rhou_s_dot_n = 0.0;
                for (int i = 0; i < 3; ++i)
                  rhou_s_dot_n += rhou_s[ibf][i] * zone_ptr->n_bf[ibf][i];

                double cf_avg_model = 0.0;
                
                cf_avg_model += 4.0*mu_lam[icv]/rho[icv]*Stn_avg[ibf];
                double direc_fac = (-0.5*mean_direc[ibf]+0.5);
                // if (rhou_s_dot_n > 0.0) {
                //   cf_avg_model += -2.0*cdel_w[ibf]*DLtn_avg[ibf];
                // } else {
                //   cf_avg_model += -2.0*cdel_w[ibf]*cdel_w[ibf]*GLtn_avg[ibf];
                // }
                cf_avg_model += direc_fac * (-2.0*cdel_w[ibf]*DLtn_avg[ibf]);
                cf_avg_model += (1.0-direc_fac) * (-2.0*cdel_w[ibf]*cdel_w[ibf]*GLtn_avg[ibf]);

                cf_avg_model *= 1.0/U_inf/U_inf;
                
                rval_dn[ibf] = (cf_avg_model - cf_ref[ibf])*(cf_avg_model - cf_ref[ibf]);

              } else {
              
                rval_dn[ibf] = 0.0;
              
              }
            
            }
          }
          return rval;
        
        }

        CtiData * local_error(vector<CtiData*>& argVec,const bool b_eval) {
          
          if (!argVec.empty()) {
            eval_cerr << "local_error() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
              
                const int icv = zone_ptr->cvobf[ibf];

                double unit_n[3];
                const double mag_n = MAG(zone_ptr->n_bf[ibf]);
                FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i]/mag_n;

                double Udotn = 0.0;
                FOR_I3 Udotn += mean_u1[ibf][i] * unit_n[i];
                double dirU[3];
                FOR_I3 dirU[i] = mean_u1[ibf][i] - Udotn * unit_n[i];
                double magU = MAG(dirU);
                FOR_I3 dirU[i] /= magU;
                double tau_wall_t = 0.0;
                FOR_I3 tau_wall_t += tau_wall_[ibf][i]*dirU[i];

                // double tau_wall_n = 0.0;
                // double tau_wall_p[3];
                // double tau_wall_t = 0.0;
                // FOR_I3 tau_wall_n += tau_wall_[ibf][i] * zone_ptr->n_bf[ibf][i];
                // FOR_I3 tau_wall_p[i] = tau_wall_[ibf][i] - tau_wall_n * zone_ptr->n_bf[ibf][i];
                // tau_wall_p[2] = 0.0; // XXX ignore spanwise component
                // tau_wall_t = MAG(tau_wall_p);
                // double sign_tau_wall_t = 1.0;
                // if (tau_wall_p[0] < 0.0) {
                //   sign_tau_wall_t = -1.0;
                // }
                // tau_wall_t *= sign_tau_wall_t;
                
                double cf_current = 2.0*tau_wall_t/rho[icv]/U_inf/U_inf;

                rval_dn[ibf] = cf_current - cf_ref[ibf];
              
              } else {
              
                rval_dn[ibf] = 0.0;
              
              }

            }
          }
          return rval;
        
        }

        CtiData * normalized_error(vector<CtiData*>& argVec,const bool b_eval) {
        
          if (!argVec.empty()) {
            eval_cerr << "normalized_error() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
              
                rval_dn[ibf] = error_avg[ibf] / cf_ref[ibf];

              } else {
              
                rval_dn[ibf] = 0.0;
              
              }

            }
          }
          return rval;

        }

        CtiData * cf_ref_func(vector<CtiData*>& argVec,const bool b_eval) {
          
          if (!argVec.empty()) {
            eval_cerr << "cf_ref() does not take args: got " << argVec.size() << endl;
            return NULL;
          }
          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);

          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            
              rval_dn[ibf] = cf_ref[ibf];
            
            }
          }
          return rval;
        }



        void initData() {
          
          // init boundary layer data exchanger
          
          if (b_blde_params) {
            assert(blde == NULL);
            blde = new BoundaryLayerDataExchanger(zone_ptr,nbl,l_bl);

            assert(bl_delta);
            assert(bl_delta_star);
            assert(bl_theta);
            assert(bl_u_edge);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              bl_delta[ibf] = 0.0;
              bl_delta_star[ibf] = 0.0;
              bl_theta[ibf] = 0.0;
              bl_u_edge[ibf] = 0.0;
            }
          }
        
          rhou_s = new double[zone_ptr->nbf][3];
          T_s    = new double[zone_ptr->nbf];

          assert(cdel_w);
          if ( !checkInit(cdel_w,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
                const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
                cdel_w[ibf] = c_slip*delta;
              } else {
                cdel_w[ibf] = 0.0;
              }
            }
          }

          assert( tau_wall_ == NULL);
          tau_wall_ = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            for (int i = 0; i < 3; ++i)
              tau_wall_[ibf][i] = 0.0;

          // place holder for parsing STATIONARY, etc for the velocity ..XXX
          
          u_wall = new double[zone_ptr->nbf][3];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            for (int i = 0; i < 3; ++i)
              u_wall[ibf][i] = 0.0;

          assert(mean_rhoun_s);
          if ( !checkInit(mean_rhoun_s,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_rhoun_s[ibf] = 0.0;
            }
            time_init = time;
          }

          assert(mean_direc);
          if ( !checkInit(mean_direc,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_direc[ibf] = 0.0;
            }
          }

          assert(mean_u1);
          if ( !checkInit(mean_u1,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv = cvobf[ibf];
              mean_u1[ibf][0] = u[icv][0];
              mean_u1[ibf][1] = 0.0;
              mean_u1[ibf][2] = 0.0;
            }
          }

          if (b_sgs_slip) {
            sgs_cdel_w = new double[zone_ptr->nbf];
            sgs_f_w = new double[zone_ptr->nbf];
            sij_w = new double[zone_ptr->nbf][6];
            betaij_w = new double[zone_ptr->nbf][6];
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              sgs_cdel_w[ibf] = 0.0;
              sgs_f_w[ibf] = 0.0;
              for (int i = 0; i < 6; ++i) {
                sij_w[ibf][i] = 0.0;
                betaij_w[ibf][i] = 0.0;
              }
            }
          }

          // read in the input reference data from file
          assert( cf_ref == NULL);
          cf_ref = new double[zone_ptr->nbf];
          assert(b_filename);
          double * x_ref_1d = NULL;
          double * cf_ref_1d = NULL;
          int n_ref_1d;
          int ierr = utils::xcol(x_ref_1d,cf_ref_1d,n_ref_1d,filename,1,2);
          if (ierr != 0) {
            CERR("cannot parse columns 1 and 2 in reference data file: " << filename);
          }
          for (int ii = 1; ii < n_ref_1d-1; ++ii) {
            if (x_ref_1d[ii] <= x_ref_1d[ii-1]) {
              CERR("x coordinate of reference data must be monotonically increasing");
            }
          }
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            const double x_bc = zone_ptr->x_bf[ibf][0];
            if (x_bc <= x_ref_1d[0]) {
              cf_ref[ibf] = cf_ref_1d[0];
            } else if (x_bc >= x_ref_1d[n_ref_1d-1]) {
              cf_ref[ibf] = cf_ref_1d[n_ref_1d-1];
            } else {
              // bisection ...
              int left = 0;
              int right = n_ref_1d-1;
              while ((right - left) > 1) {
                const int middle = (left + right)/2; // equivalent to floor...
                if (x_bc >= x_ref_1d[middle])
                  left = middle;
                else
                  right = middle;
              }
              const double w0 = (x_ref_1d[right]-x_bc)/(x_ref_1d[right]-x_ref_1d[left]);
              cf_ref[ibf] = w0*cf_ref_1d[left] + (1.0-w0)*cf_ref_1d[right];
            }
          }
          delete[] x_ref_1d;
          delete[] cf_ref_1d;

          // gradient data
          assert(GLtn_avg);
          if ( !checkInit(GLtn_avg,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              GLtn_avg[ibf] = 0.0;
            }
          }

          assert(DLtn_avg);
          if ( !checkInit(DLtn_avg,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              DLtn_avg[ibf] = 0.0;
            }
          }

          assert(Stn_avg);
          if ( !checkInit(Stn_avg,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
                const int icv = zone_ptr->cvobf[ibf];
                const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
                const double magu = MAG(u[icv]);
                Stn_avg[ibf] = magu / delta;
              } else {
                Stn_avg[ibf] = 0.0;
              }
            }
          }

          // model gradient data
          if (b_modelgrad) {
            assert(gradJmodel == NULL);
            assert(hessJmodel == NULL);
            gradJmodel = new double[zone_ptr->nbf];
            hessJmodel = new double[zone_ptr->nbf];
          }

          // partition the domain
          // 1. figure out the global Lx
          double my_min_x = BIG_DOUBLE;
          double my_max_x = -BIG_DOUBLE;
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            const double this_x = zone_ptr->x_bf[ibf][0];
            if ( this_x < my_min_x) {
              my_min_x = this_x;
            }
            if ( this_x > my_max_x) {
              my_max_x = this_x;
            }
          }
          double my_buf[2] = {-my_min_x,my_max_x};
          double buf[2];
          MPI_Allreduce(my_buf,buf,2,MPI_DOUBLE,MPI_MAX,mpi_comm);
          double min_x = -buf[0];
          double max_x = buf[1];
          const double Lx = max_x - min_x;
          assert(Lx > 0.0);
          if (mpi_rank == 0) cout << "WmOptSlipBc: found Lx = " << Lx << endl;
          // 2. compute the number of elements
          nfe_bc = (int)(ceil((Lx / dx_fe) + 1.0));
          if (dx_fe > Lx) { // single element
            nfe_bc = 1;
          }
          assert(nfe_bc >= 1);
          if (mpi_rank == 0) cout << "WmOptSlipBc: computed nfe_bc = " << nfe_bc << endl;
          // 3. build xfe_bc
          xfe_bc = new double[nfe_bc];
          //double step_size = Lx / (double(nfe_bc-1)); //XXX
          for (int i = 0; i < nfe_bc; ++i) {
            xfe_bc[i] = min_x + double(i)/double(nfe_bc-1) * Lx;
          }
          if (mpi_rank == 0) { //XXX
            cout << "> DEBUG: x_fe = [" << xfe_bc[0];
            for (int i = 1; i < nfe_bc; ++i) cout << ", " << xfe_bc[i];
            cout << "]" << endl;
          }
          // 4. prefill cslip_fe
          cslip_fe = new double[nfe_bc];
          for (int i = 0; i < nfe_bc; ++i) {
            //cslip_fe[i] = c_slip * (1.0 + 0.1*sin(xfe_bc[i]*3.14159265/Lx));
            cslip_fe[i] = c_slip;
          }

          // 5. precompute weights
          feobf = new int[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            // search for bounding interval of x_bf in xfe_bc
            double this_x = zone_ptr->x_bf[ibf][0];
            int j_lo = 0;
            int j_hi = nfe_bc-1;
            while ((j_hi - j_lo) > 1) {
              const int j_mid = (j_lo + j_hi)/2;
              if (xfe_bc[j_mid] <= this_x) {
                j_lo = j_mid;
              } else {
                j_hi = j_mid;
              }
            }
            //assert(xfe_bc[j_lo] <= this_x);
            //assert(xfe_bc[j_lo+1] >= this_x);
            feobf[ibf] = j_lo;
          }

          // 5.5. model gradients for elements
          if (b_modelgrad) {
            gradJmodel_fe = new double[nfe_bc];
            hessJmodel_fe = new double[nfe_bc];
          }

          // 6. check for optim restart data
          bool bRestart = false;
          double * restart_cslip = NULL;
          int nrestart = readOptimState(restart_cslip);
          MPI_Bcast(&nrestart,1,MPI_INT,0,mpi_comm);
          bRestart = (nrestart > 0);
          if (bRestart) {

            if (mpi_rank == 0) assert(restart_cslip != NULL);
            if (mpi_rank != 0) restart_cslip = new double[nrestart];
            MPI_Bcast(restart_cslip,nrestart,MPI_DOUBLE,0,mpi_comm);

            if (nrestart == nfe_bc) { // no interpolation needed
              for (int ife = 0; ife < nfe_bc; ++ife) {
                cslip_fe[ife] = restart_cslip[ife];
              }
            } else if (nrestart == 1) {
              for (int ife = 0; ife < nfe_bc; ++ife) {
                cslip_fe[ife] = restart_cslip[0];
              }
            } else { // interpolation required
              int jfe = 0;
              cslip_fe[0] = restart_cslip[0];
              for (int ife = 1; ife < nfe_bc; ++ife) {
                double chi = double(ife)/double(nfe_bc-1);
                double eta = double(jfe)/double(nrestart-1);
                while (chi > eta) {
                  ++jfe;
                  eta = double(jfe)/double(nrestart-1);
                }
                --jfe;
                eta = double(jfe)/double(nrestart-1);
                double eta_next = double(jfe+1)/double(nrestart-1);
                // linear interpolation
                cslip_fe[ife]  = restart_cslip[jfe]*(eta_next-chi)/(eta_next-eta);
                cslip_fe[ife] += restart_cslip[jfe+1]*(chi-eta)/(eta_next-eta);
                // // nearest neighbor
                // if (abs(eta_next-chi) < abs(eta-chi) && (jfe+1) < nrestart) {
                //   cslip_fe[ife] = restart_cslip[jfe+1];
                // } else {
                //   cslip_fe[ife] = restart_cslip[jfe];
                // }
              }
              if (mpi_rank == 0) {
                cout << " > DEBUG: cslip_fe interpolation: " << endl;
                cout << " > cslip_fe = [" << cslip_fe[0];
                for (int ife = 1; ife < nfe_bc; ++ife) {
                  cout << ", " << cslip_fe[ife];
                }
                cout << "]" << endl;
                cout << " > cslip_restart = [" << restart_cslip[0];
                for (int jfe = 1; jfe < nrestart; ++jfe) {
                  cout << ", " << restart_cslip[jfe];
                }
                cout << "]" << endl;
              }
            }

            if (restart_cslip != NULL) delete[] restart_cslip;

          }
          // 7. update local slip lengths
          updateLocalSlipLength();

          prev_optim_cslip = 0.0;

          if ( ife_optim == BIG_INT) {
            ife_optim = rand() % nfe_bc; // choose a random starting fe to optimize
          } else {
            ife_optim %= nfe_bc;
          }
          MPI_Bcast(&ife_optim,1,MPI_INT,0,mpi_comm);

          optim_prev_time = time;

          error_avg = new double[zone_ptr->nbf];
          error_counts = new double[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            error_avg[ibf] = 0.0;
            error_counts[ibf] = 0.0;
          }

          cf_avg = new double[zone_ptr->nbf];
          cf_counts = new double[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            cf_avg[ibf] = 0.0;
            cf_counts[ibf] = 0.0;
          }

          if (b_eps_net) {
            Lobj_eps_net = new double[zone_ptr->nbf];
            idx_eps_net = 0;
            optim_state = EPS_NET;
            const double w0 = double(idx_eps_net) / double(n_eps_net-1);
            cslip_prev_eps_net = cslip_fe[ife_optim];
            cslip_fe[ife_optim] = w0*eps_net_max + (1.0-w0)*eps_net_min;
            updateLocalSlipLength();
          }

          if (b_modelgrad) {
            optim_state = GRAD_MODEL;
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( rhou_s_d == NULL);
          assert( T_s_d    == NULL);
          assert( cdel_w_d == NULL);
          assert( u_wall_d == NULL);
          assert( tau_wall_d_ == NULL);

          assert( cf_ref_d   == NULL);
          assert( mean_rhoun_s_d == NULL);
          assert( mean_direc_d == NULL);
          assert( mean_u1_d  == NULL);
          assert( GLtn_avg_d == NULL);
          assert( DLtn_avg_d == NULL);
          assert( Stn_avg_d  == NULL);

          assert( gradJmodel_d == NULL);
          assert( hessJmodel_d == NULL);

          assert( error_avg_d == NULL);
          assert( error_counts_d == NULL);
          assert( cf_avg_d == NULL);
          assert( cf_counts_d == NULL);

          cti::utils::initOnDevice(rhou_s_d,3*zone_ptr->nbf);
          cti::utils::initOnDevice(T_s_d   ,  zone_ptr->nbf);

          // set an initial guess for the cdel_w .. will be updated
          cti::utils::initAndCopyToDevice(cdel_w_d,cdel_w,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_wall_d,(double*)u_wall,3*zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(tau_wall_d_,(double*)tau_wall_,3*zone_ptr->nbf);

          if (b_sgs_slip) {
            cti::utils::initAndCopyToDevice(sgs_cdel_w_d,sgs_cdel_w,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(sgs_f_w_d,sgs_f_w,zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(sij_w_d,(double*)sij_w,6*zone_ptr->nbf);
            cti::utils::initAndCopyToDevice(betaij_w_d,(double*)betaij_w,6*zone_ptr->nbf);
          }

          // set cf_ref
          cti::utils::initAndCopyToDevice(cf_ref_d,cf_ref,zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(mean_rhoun_s_d,mean_rhoun_s,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(mean_direc_d,mean_direc,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(mean_u1_d,(double*)mean_u1,3*zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(GLtn_avg_d,GLtn_avg,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(DLtn_avg_d,DLtn_avg,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(Stn_avg_d ,Stn_avg ,zone_ptr->nbf);

          if (b_modelgrad) {
            cti::utils::initOnDevice(gradJmodel_d,zone_ptr->nbf);
            cti::utils::initOnDevice(hessJmodel_d,zone_ptr->nbf);
          }

          cti::utils::initAndCopyToDevice(error_avg_d, error_avg, zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(error_counts_d, error_counts, zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(cf_avg_d, cf_avg, zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(cf_counts_d, cf_counts, zone_ptr->nbf);

          cti::utils::initOnDevice(res_d,1024);
          
#endif

        }

        // write the optimizer state for restart
        void writeOptimState() {
          if (mpi_rank == 0) {
            char filename[128];
            sprintf(filename,"optim_state.%s.dat",zone_ptr->name.c_str());
            utils::mkdir_for_file(filename);
            std::ofstream ofp;
            ofp.open(filename,std::ofstream::out|std::ofstream::trunc);
            if (!ofp.is_open()) {
                cout << "WARNING: cannot open optimizer state file: skipping..." << endl;
            }
            else {
              ofp << nfe_bc << endl;
              for (int ife = 0; ife < nfe_bc; ++ife) {
                if (ife == ife_optim &&
                    (optim_state == EVAL_P2 || 
                     optim_state == TERMINAL) ) {
                  ofp << cslip_fe[ife]-DC_STEP << endl;

                } else if (ife == ife_optim && optim_state == EPS_NET) {
                  ofp << cslip_prev_eps_net << endl;

                } else { // EVAL_P1, GRAD_MODEL, ife != ife_optim
                    ofp << cslip_fe[ife] << endl;

                }
              }
              ofp.close();
            }
          }
        }

        // read the optimizer state for restart
        // returns the length of vector ret
        int readOptimState(double *& ret) {
          if (mpi_rank == 0) {
            char filename[128];
            sprintf(filename,"optim_state.%s.dat",zone_ptr->name.c_str());
            if (!utils::fileExists(filename)) {
              ret = NULL;
              return 0;
            }
            std::ifstream ifp;
            ifp.open(filename,std::ifstream::in);
            if (!ifp.is_open()) {
              cout << "WARNING: cannot open optimizer state file: skipping..." << endl;
              return 0;
            }
            else {
              int nn, ii;
              ifp >> nn;
              ret = new double[nn];
              double val;
              ii = 0;
              while(ifp >> val) {
                ret[ii] = val;
                ii++;
              }
              if (ii != nn) {
                cout << "WARNING: optim_state.dat is inconsistent: skipping..." << endl;
                ret = NULL;
                return 0;
              }
              return nn;
            }
          }

          return 0;
        }

        // update slip length on bfs from fes
        void updateLocalSlipLength() {

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          
            const double y1 = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];
            const int ife_bc = feobf[ibf];
            const double this_x = zone_ptr->x_bf[ibf][0];

            if (nfe_bc == 1) {
              cdel_w[ibf] = y1 * cslip_fe[ife_bc];
              continue;
            }

            const double w1 = (this_x - xfe_bc[ife_bc])/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
            const double w0 = (xfe_bc[ife_bc+1] - this_x)/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
            if (w0 < 0.0) {
              cout << " > DEBUG: w0 < 0.0 - x " << this_x << " xfe " << xfe_bc[ife_bc] << endl;
              cout.flush();
            }
            if (w1 < 0.0) {
              cout << " > DEBUG: w1 < 0.0 - x " << this_x << " xfe " << xfe_bc[ife_bc] << endl;
              cout.flush();
            }
            if (w0 > 1.0) {
              cout << " > DEBUG: w0 > 1.0 - x " << this_x << " xfe " << xfe_bc[ife_bc] << endl;
              cout.flush();
            }
            if (w1 > 1.0) {
              cout << " > DEBUG: w1 > 1.0 - x " << this_x << " xfe " << xfe_bc[ife_bc] << endl;
              cout.flush();
            }
            assert(w0 >= 0.0 - 1e-8);
            assert(w1 >= 0.0 - 1e-8);
            assert(w0 <= 1.0 + 1e-8);
            assert(w1 <= 1.0 + 1e-8);
            const double cslip_avg = w0 * cslip_fe[ife_bc] + w1 * cslip_fe[ife_bc+1];
            if (cslip_avg < 0.0) {
              cout << "assert(cslip_avg >= 0.0) failed: " << this_x << xfe_bc[ife_bc] 
                   << xfe_bc[ife_bc+1] << cslip_fe[ife_bc] << cslip_fe[ife_bc+1] << endl;
            }
            assert(cslip_avg >= 0.0);
            cdel_w[ibf] = y1 * cslip_avg;
          
          }

        }

        void gradientUpdateSlipLength() {

          if (step%check_interval == 0) {
            evaluateObjectiveNoReset();

            if (mpi_rank == 0) {
              cout << " > WmOptSlipIso: check interval" << endl;
              cout << " > Lobj = " << Lobj << endl;
              cout << " > L2 = " << l2_err << endl;
              cout << " > time = " << time << endl;
            }
          }

          // only update as often as requested
          if ((time - optim_prev_time) < dt_optim_interval) {return;}

          // we want to update now ...
          optim_prev_time = time;
          evaluateObjectiveAndReset();

          if (mpi_rank == 0) {
            cout << " > WmOptSlipIso: performing gradient update of slip length" << endl;
            cout << " > Lobj = " << Lobj << endl;
            cout << " > Lobj_prev = " << Lobj_prev << endl;
            cout << " > ife_optim = " << ife_optim << endl;
            cout << " > cslip_fe[ife_optim] = " << cslip_fe[ife_optim] << endl;
            cout << " > prev_optim_cslip = " << prev_optim_cslip << endl;
            cout << " > optim_state = " << optim_state << endl;
            cout << " > time = " << time << endl;
            cout << " > cf_area_avg = " << cf_area_avg << endl;
            cout << " > cf_ref_avg = " << cf_ref_avg << endl;
            cout << " > L2 = " << l2_err << endl;
          }
        
          if (optim_state == EVAL_P1) {
            // we are currently evaluating the first point in the finite difference
            prev_optim_cslip = cslip_fe[ife_optim];
            cslip_fe[ife_optim] += DC_STEP;
            Lobj_prev = Lobj;

            // check if next optimization step should be terminal
            if (n_optim_steps >= max_optim_steps-1) {
              optim_state = TERMINAL;
            } else {
              optim_state = EVAL_P2;
            }

          } else if (optim_state == EVAL_P2) {
            // we have evaluated at least one other point previously
            double this_cslip = cslip_fe[ife_optim];
            double grad_approx = (Lobj - Lobj_prev)/(this_cslip - prev_optim_cslip);

            // secant method update
            double next_cslip = prev_optim_cslip - step_optim * grad_approx;
            prev_optim_cslip = this_cslip;
            cslip_fe[ife_optim] = next_cslip;

            Lobj_prev = Lobj;

            n_optim_steps++;
            
            // handle clipping (C_slip \in [0.0,1.5])
            if (cslip_fe[ife_optim] < 0.0) {
              cslip_fe[ife_optim] = 0.0;
            } else if (cslip_fe[ife_optim] > 1.5) {
              cslip_fe[ife_optim] = 1.5;
            }

            optim_state = EVAL_P1; // go back to grad evaluation
            
          } else if (optim_state == TERMINAL) {
            // this is the last update with this element
            double this_cslip = cslip_fe[ife_optim];
            double grad_approx = (Lobj - Lobj_prev)/(this_cslip - prev_optim_cslip);

            // approx gradient update
            double next_cslip = prev_optim_cslip - step_optim * grad_approx;
            cslip_fe[ife_optim] = next_cslip;

            Lobj_prev = Lobj;

            n_optim_steps = 0;
          
            // handle clipping (C_slip \in [0.0,1.5])
            if (cslip_fe[ife_optim] < 0.0) {
              cslip_fe[ife_optim] = 0.0;
            } else if (cslip_fe[ife_optim] > 1.5) {
              cslip_fe[ife_optim] = 1.5;
            }
            
            // // choose another distinct element at random
            // if (mpi_rank == 0) {
            //   int ife_next = ife_optim;
            //   while (ife_next == ife_optim && nfe_bc > 1) {
            //     ife_next = rand() % nfe_bc;
            //   }
            //   assert(ife_next >= 0);
            //   assert(ife_next < nfe_bc);
            //   assert(ife_next != ife_optim || nfe_bc == 1);
            //   ife_optim = ife_next;
            // }

            // // choose another distinct element based on arg l_inf error
            // if (mpi_rank == 0) {
            //   if ( rand() % 100 < 0) {
            //     ife_optim = ife_max_error;
            //   } else {
            //     int ife_next = ife_optim;
            //     while (ife_next == ife_optim && nfe_bc > 1) {
            //       ife_next = rand() % nfe_bc;
            //     }
            //     ife_optim = ife_next;
            //   }
            // }
            // MPI_Bcast(ife_optim,1,MPI_INT,0,mpi_comm);

            // scanning
            ++(ife_optim);
            if (ife_optim >= nfe_bc) ife_optim = 0;

            prev_optim_cslip = cslip_fe[ife_optim]; // want prev. from new chosen fe

            if (b_eps_net && nfe_bc > 1) {
              optim_state = EPS_NET;
              idx_eps_net = 0;
              const double w0 = double(idx_eps_net) / double(n_eps_net-1);
              cslip_prev_eps_net = cslip_fe[ife_optim];
              cslip_fe[ife_optim] = w0*eps_net_max + (1.0-w0)*eps_net_min;
            } else {
              optim_state = EVAL_P1;
            }
            
          } else if (optim_state == EPS_NET) {
            // we are currently performing a grid search
            if ( idx_eps_net < n_eps_net-1) {

              Lobj_eps_net[idx_eps_net] = Lobj;

              idx_eps_net++;

              const double w0 = double(idx_eps_net) / double(n_eps_net-1);
              cslip_fe[ife_optim] = w0*eps_net_max + (1.0-w0)*eps_net_min;

              // don't change the optim state

            } else { // last step of grid search

              Lobj_eps_net[idx_eps_net] = Lobj; // save the last data point

              assert(idx_eps_net == n_eps_net-1);
              double min_Lobj = Lobj_eps_net[0];
              int min_idx_eps_net = 0;
              for (int j = 1; j < n_eps_net; ++j) {
                if (Lobj_eps_net[j] < min_Lobj) {
                  min_Lobj = Lobj_eps_net[j];
                  min_idx_eps_net = j;
                }
              }
              idx_eps_net = min_idx_eps_net;
              const double w0 = double(idx_eps_net) / double(n_eps_net-1);
              cslip_fe[ife_optim] = w0*eps_net_max + (1.0-w0)*eps_net_min;

              // switch to gradient descent
              optim_state = EVAL_P1;

            }

          } else if (optim_state == GRAD_MODEL) {

            // we have a full gradient model so update every element
            for (int ife = 0; ife < nfe_bc; ++ife) {
              cslip_fe[ife] -= gradJmodel[ife]/hessJmodel[ife];
              if (cslip_fe[ife] < 0.0) {
                cslip_fe[ife] = 0.0;
              }
              if (cslip_fe[ife] > 1.5) {
                cslip_fe[ife] = 1.5;
              }
            }

          } else {
            if (mpi_rank == 0) cout << " > WmOptSlipBc: unknown optim_state: " 
                                    << optim_state << endl;
            assert(0);

          }

          if (mpi_rank == 0) writeOptimState();

          if (mpi_rank == 0) cout << " > WmOptSlipBc: updated cslip: " << cslip_fe[ife_optim] << endl;


          updateLocalSlipLength();

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToDevice(cdel_w_d, cdel_w, zone_ptr->nbf);

#endif

        }

        // evaluate the objective function and reset averages and counts
        void evaluateObjectiveAndReset() {

          if (mpi_rank == 0) cout << " > WmOptSlipBc: evaluating objective" << endl;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost(error_avg, error_avg_d, zone_ptr->nbf);
          cti::utils::copyToHost(cf_avg, cf_avg_d, zone_ptr->nbf);
          cti::utils::copyToHost(error_counts, error_counts_d, zone_ptr->nbf);
          cti::utils::copyToHost(cf_counts, cf_counts_d, zone_ptr->nbf);
          
          if (b_modelgrad) {
            cti::utils::copyToHost(gradJmodel, gradJmodel_d, zone_ptr->nbf);
            cti::utils::copyToHost(hessJmodel, hessJmodel_d, zone_ptr->nbf);
          }

#endif
         
          // find the element with the max error
          ife_max_error = 0;
          if (nfe_bc > 1) {
            int my_max_loc = utils::getMaxLoc(error_avg,zone_ptr->nbf);
            if (my_max_loc >= 0) { // we own the max loc
              ife_max_error = feobf[my_max_loc];
            } else { // we don't own the max loc
              ife_max_error = -1;
            }
            MPI_Allreduce(MPI_IN_PLACE,&ife_max_error,1,MPI_INT,MPI_MAX,mpi_comm);
          }

          cti::utils::dumpRange(error_avg,zone_ptr->nbf,zone_ptr->getName()+":error_avg");
          cti::utils::dumpRange(cf_avg,zone_ptr->nbf,zone_ptr->getName()+":cf_avg");

          // reduce the global error
          double *my_Lobj_arr = new double[nfe_bc];
          double *my_area_arr = new double[nfe_bc];
          double *my_cfref_arr = new double[nfe_bc];
          double my_l2_err = 0.0;
          for (int ife = 0; ife < nfe_bc; ++ife) {
            my_Lobj_arr[ife] = 0.0;
            my_area_arr[ife] = 0.0;
            my_cfref_arr[ife] = 0.0;
          }
          if (nfe_bc == 1) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              my_Lobj_arr[0] += error_avg[ibf] * zone_ptr->area_bf[ibf];
              my_area_arr[0] += zone_ptr->area_bf[ibf];
              my_cfref_arr[0] += zone_ptr->area_bf[ibf];

              // also compute L2 error
              my_l2_err += error_avg[ibf] * error_avg[ibf] * zone_ptr->area_bf[ibf];

              error_avg[ibf] = 0.0;
              error_counts[ibf] = 0.0;
            }
          } else {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int ife_bc = feobf[ibf];
              const double this_x = zone_ptr->x_bf[ibf][0];
              
              const double w1 = (this_x - xfe_bc[ife_bc])/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
              const double w0 = (xfe_bc[ife_bc+1] - this_x)/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);

              my_Lobj_arr[ife_bc] += w0 * error_avg[ibf] * zone_ptr->area_bf[ibf];
              my_Lobj_arr[ife_bc+1] += w1 * error_avg[ibf] * zone_ptr->area_bf[ibf];

              my_area_arr[ife_bc] += w0 * zone_ptr->area_bf[ibf];
              my_area_arr[ife_bc+1] += w1 * zone_ptr->area_bf[ibf];

              my_cfref_arr[ife_bc] += w0 * cf_ref[ibf] * zone_ptr->area_bf[ibf];
              my_cfref_arr[ife_bc+1] += w1 * cf_ref[ibf] * zone_ptr->area_bf[ibf];
              
              // also compute L2 error
              my_l2_err += error_avg[ibf] * error_avg[ibf] * zone_ptr->area_bf[ibf];
            }
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              error_avg[ibf] = 0.0;
              error_counts[ibf] = 0.0;
            }
          }

          double *Lobj_arr = new double[nfe_bc];
          double *area_arr = new double[nfe_bc];
          double *cfref_arr = new double[nfe_bc];

          MPI_Allreduce(my_Lobj_arr,Lobj_arr,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
          MPI_Allreduce(my_area_arr,area_arr,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
          MPI_Allreduce(my_cfref_arr,cfref_arr,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
          MPI_Allreduce(&my_l2_err,&l2_err,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          l2_err /= zone_ptr->area_global;
          l2_err = sqrt(l2_err);

          Lobj = 0.0;
          for (int ife = 0; ife < nfe_bc; ++ife) {
            Lobj_arr[ife] /= area_arr[ife];
            Lobj += Lobj_arr[ife]*Lobj_arr[ife] * area_arr[ife];
          }
          Lobj *= 0.5;
          Lobj /= zone_ptr->area_global;

          for (int ife = 0; ife < nfe_bc; ++ife) {
            cfref_arr[ife] /= area_arr[ife];
          }

          if (mpi_rank == 0) {
            cout << "DEBUG: cf_ref[ife] = [" << cfref_arr[0];
            for (int ife=1; ife < nfe_bc; ++ife) {
              cout << ", " << cfref_arr[ife];
            }
            cout << "]" << endl;
          }
          
          // double my_Lobj = 0.0;
          // for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          //   my_Lobj += error_avg[ibf]*error_avg[ibf]*zone_ptr->area_bf[ibf];
          //   // reset average and counts
          //   error_avg[ibf] = 0.0;
          //   error_counts[ibf] = 0.0;
          // }
          // Lobj = 0.0;
          // MPI_Allreduce(&my_Lobj,&Lobj,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          // Lobj *= 0.5;
          // Lobj /= zone_ptr->area_global;
          
          double my_cf_area_avg = 0.0;
          double my_cf_ref_avg = 0.0;
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            my_cf_area_avg += cf_avg[ibf]*zone_ptr->area_bf[ibf];
            my_cf_ref_avg += cf_ref[ibf]*zone_ptr->area_bf[ibf];
            // reset average and counts
            cf_avg[ibf] = 0.0;
            cf_counts[ibf] = 0.0;
          }
          MPI_Allreduce(&my_cf_area_avg,&cf_area_avg,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          cf_area_avg /= zone_ptr->area_global;
          MPI_Allreduce(&my_cf_ref_avg,&cf_ref_avg,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          cf_ref_avg /= zone_ptr->area_global;

          // Normalize the error
          Lobj /= cf_ref_avg;

          if (b_modelgrad) {
            // store the model gradients for the fe's

            double * area_norm = new double[nfe_bc];
            for (int ife = 0; ife < nfe_bc; ++ife) {
              gradJmodel_fe[ife] = 0.0;
              hessJmodel_fe[ife] = 0.0;
              area_norm[ife] = 0.0;
            }

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int ife_bc = feobf[ibf];
              const double this_x = zone_ptr->x_bf[ibf][0];
              
              if (nfe_bc == 1) {
                gradJmodel_fe[0] += gradJmodel[ibf];
                hessJmodel_fe[0] += hessJmodel[ibf];
                continue;
              }

              const double w1 = (this_x - xfe_bc[ife_bc])/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
              const double w0 = (xfe_bc[ife_bc+1] - this_x)/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
              gradJmodel_fe[ife_bc] += w0 * gradJmodel[ibf] * zone_ptr->area_bf[ibf];
              gradJmodel_fe[ife_bc+1] += w1 * gradJmodel[ibf] * zone_ptr->area_bf[ibf];
              hessJmodel_fe[ife_bc] += w0 * hessJmodel[ibf] * zone_ptr->area_bf[ibf];
              hessJmodel_fe[ife_bc+1] += w1 * hessJmodel[ibf] * zone_ptr->area_bf[ibf];
              area_norm[ife_bc] += w0 * zone_ptr->area_bf[ibf];
              area_norm[ife_bc+1] += w1 * zone_ptr->area_bf[ibf];
            }

            MPI_Allreduce(MPI_IN_PLACE,gradJmodel_fe,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
            MPI_Allreduce(MPI_IN_PLACE,hessJmodel_fe,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
            MPI_Allreduce(MPI_IN_PLACE,area_norm,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
            
            for (int ife = 0; ife < nfe_bc; ++ife) {
              gradJmodel_fe[ife] /= area_norm[ife];
              hessJmodel_fe[ife] /= area_norm[ife];
            }

            delete[] area_norm;
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToDevice(error_avg_d, error_avg, zone_ptr->nbf);
          cti::utils::copyToDevice(error_counts_d, error_counts, zone_ptr->nbf);

          cti::utils::copyToDevice(cf_avg_d, cf_avg, zone_ptr->nbf);
          cti::utils::copyToDevice(cf_counts_d, cf_counts, zone_ptr->nbf);

#endif

        }
        
        // evaluate the objective function without resetting
        void evaluateObjectiveNoReset() {

          if (mpi_rank == 0) cout << " > WmOptSlipBc: evaluating objective (no reset)" << endl;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost(error_avg, error_avg_d, zone_ptr->nbf);
          cti::utils::copyToHost(cf_avg, cf_avg_d, zone_ptr->nbf);
          cti::utils::copyToHost(error_counts, error_counts_d, zone_ptr->nbf);
          cti::utils::copyToHost(cf_counts, cf_counts_d, zone_ptr->nbf);
          
          if (b_modelgrad) {
            cti::utils::copyToHost(gradJmodel, gradJmodel_d, zone_ptr->nbf);
            cti::utils::copyToHost(hessJmodel, hessJmodel_d, zone_ptr->nbf);
          }

#endif
         
          // find the element with the max error
          ife_max_error = 0;
          if (nfe_bc > 1) {
            int my_max_loc = utils::getMaxLoc(error_avg,zone_ptr->nbf);
            if (my_max_loc >= 0) { // we own the max loc
              ife_max_error = feobf[my_max_loc];
            } else { // we don't own the max loc
              ife_max_error = -1;
            }
            MPI_Allreduce(MPI_IN_PLACE,&ife_max_error,1,MPI_INT,MPI_MAX,mpi_comm);
          }

          cti::utils::dumpRange(error_avg,zone_ptr->nbf,zone_ptr->getName()+":error_avg");
          cti::utils::dumpRange(cf_avg,zone_ptr->nbf,zone_ptr->getName()+":cf_avg");

          // reduce the global error
          double *my_Lobj_arr = new double[nfe_bc];
          double *my_area_arr = new double[nfe_bc];
          double *my_cfref_arr = new double[nfe_bc];
          double my_l2_err = 0.0;
          for (int ife = 0; ife < nfe_bc; ++ife) {
            my_Lobj_arr[ife] = 0.0;
            my_area_arr[ife] = 0.0;
            my_cfref_arr[ife] = 0.0;
          }
          if (nfe_bc == 1) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              my_Lobj_arr[0] += error_avg[ibf] * zone_ptr->area_bf[ibf];
              my_area_arr[0] += zone_ptr->area_bf[ibf];
              my_cfref_arr[0] += zone_ptr->area_bf[ibf];

              // also compute L2 error
              my_l2_err += error_avg[ibf] * error_avg[ibf] * zone_ptr->area_bf[ibf];
            }
          } else {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int ife_bc = feobf[ibf];
              const double this_x = zone_ptr->x_bf[ibf][0];
              
              const double w1 = (this_x - xfe_bc[ife_bc])/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
              const double w0 = (xfe_bc[ife_bc+1] - this_x)/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);

              my_Lobj_arr[ife_bc] += w0 * error_avg[ibf] * zone_ptr->area_bf[ibf];
              my_Lobj_arr[ife_bc+1] += w1 * error_avg[ibf] * zone_ptr->area_bf[ibf];

              my_area_arr[ife_bc] += w0 * zone_ptr->area_bf[ibf];
              my_area_arr[ife_bc+1] += w1 * zone_ptr->area_bf[ibf];

              my_cfref_arr[ife_bc] += w0 * cf_ref[ibf] * zone_ptr->area_bf[ibf];
              my_cfref_arr[ife_bc+1] += w1 * cf_ref[ibf] * zone_ptr->area_bf[ibf];
              
              // also compute L2 error
              my_l2_err += error_avg[ibf] * error_avg[ibf] * zone_ptr->area_bf[ibf];
            }
          }

          double *Lobj_arr = new double[nfe_bc];
          double *area_arr = new double[nfe_bc];
          double *cfref_arr = new double[nfe_bc];

          MPI_Allreduce(my_Lobj_arr,Lobj_arr,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
          MPI_Allreduce(my_area_arr,area_arr,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
          MPI_Allreduce(my_cfref_arr,cfref_arr,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
          MPI_Allreduce(&my_l2_err,&l2_err,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          l2_err /= zone_ptr->area_global;
          l2_err = sqrt(l2_err);

          Lobj = 0.0;
          for (int ife = 0; ife < nfe_bc; ++ife) {
            Lobj_arr[ife] /= area_arr[ife];
            Lobj += Lobj_arr[ife]*Lobj_arr[ife] * area_arr[ife];
          }
          Lobj *= 0.5;
          Lobj /= zone_ptr->area_global;

          for (int ife = 0; ife < nfe_bc; ++ife) {
            cfref_arr[ife] /= area_arr[ife];
          }

          double my_cf_area_avg = 0.0;
          double my_cf_ref_avg = 0.0;
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            my_cf_area_avg += cf_avg[ibf]*zone_ptr->area_bf[ibf];
            my_cf_ref_avg += cf_ref[ibf]*zone_ptr->area_bf[ibf];
          }
          MPI_Allreduce(&my_cf_area_avg,&cf_area_avg,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          cf_area_avg /= zone_ptr->area_global;
          MPI_Allreduce(&my_cf_ref_avg,&cf_ref_avg,1,MPI_DOUBLE,MPI_SUM,mpi_comm);
          cf_ref_avg /= zone_ptr->area_global;

          // Normalize the error
          Lobj /= cf_ref_avg;

          if (b_modelgrad) {
            // store the model gradients for the fe's

            double * area_norm = new double[nfe_bc];
            for (int ife = 0; ife < nfe_bc; ++ife) {
              gradJmodel_fe[ife] = 0.0;
              hessJmodel_fe[ife] = 0.0;
              area_norm[ife] = 0.0;
            }

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int ife_bc = feobf[ibf];
              const double this_x = zone_ptr->x_bf[ibf][0];
              
              if (nfe_bc == 1) {
                gradJmodel_fe[0] += gradJmodel[ibf];
                hessJmodel_fe[0] += hessJmodel[ibf];
                continue;
              }

              const double w1 = (this_x - xfe_bc[ife_bc])/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
              const double w0 = (xfe_bc[ife_bc+1] - this_x)/(xfe_bc[ife_bc+1] - xfe_bc[ife_bc]);
              gradJmodel_fe[ife_bc] += w0 * gradJmodel[ibf] * zone_ptr->area_bf[ibf];
              gradJmodel_fe[ife_bc+1] += w1 * gradJmodel[ibf] * zone_ptr->area_bf[ibf];
              hessJmodel_fe[ife_bc] += w0 * hessJmodel[ibf] * zone_ptr->area_bf[ibf];
              hessJmodel_fe[ife_bc+1] += w1 * hessJmodel[ibf] * zone_ptr->area_bf[ibf];
              area_norm[ife_bc] += w0 * zone_ptr->area_bf[ibf];
              area_norm[ife_bc+1] += w1 * zone_ptr->area_bf[ibf];
            }

            MPI_Allreduce(MPI_IN_PLACE,gradJmodel_fe,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
            MPI_Allreduce(MPI_IN_PLACE,hessJmodel_fe,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
            MPI_Allreduce(MPI_IN_PLACE,area_norm,nfe_bc,MPI_DOUBLE,MPI_SUM,mpi_comm);
            
            for (int ife = 0; ife < nfe_bc; ++ife) {
              gradJmodel_fe[ife] /= area_norm[ife];
              hessJmodel_fe[ife] /= area_norm[ife];
            }

            delete[] area_norm;
          }

        }
        
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {

          // recall rk_stage is 1,2,3...

          assert((rk_stage >= 1)&&(rk_stage <= NRK));

          gradientUpdateSlipLength(); // performs gradient update at time intervals

          // add the flux...

          const double gogm1 = gamma/(gamma-1.0);

          const double t_c = min(abs(t_avg),time-time_init);
          const double eps_t = dt/(dt+t_c);

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          
            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
            
              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

              // double tau_wall_mag = 0.0;
              // for (int i = 0; i < 3; ++i)
              //   tau_wall_mag += tau_wall_[ibf][i]*tau_wall_[ibf][i];
              // if (tau_wall_mag > 0.0)
              //   tau_wall_mag = sqrt(tau_wall_mag);
              // else
              //   tau_wall_mag = 0.0;

              // // compute the slip coefficient state
              // double cslip_state = cdel_w[ibf] / delta;
              // cslip_state = cslip_state*cslip_state;

              // // compute the slip length gradient
              // double grad_cslip2 = 0.0;
              // //grad_cslip2 += 4.0*mu_lam[icv]*Stn_avg[ibf]/rho[icv]/U_inf/U_inf;
              // //grad_cslip2 += -2.0*cslip_state*delta*delta*GLtn_avg[ibf]/U_inf/U_inf;
              // grad_cslip2 += 2.0*tau_wall_mag/rho[icv]/U_inf/U_inf;
              // grad_cslip2 += -cf_ref[ibf];
              // grad_cslip2 *= -2.0*GLtn_avg[ibf];

              // // compute the gradient step size
              // double inv_step_size = 4.0*delta*delta;
              // inv_step_size *= GLtn_avg[ibf]*GLtn_avg[ibf]/U_inf/U_inf;

              // // update the slip length...
              // // relaxed gradient update (relaxation of ~= dt/T_avg)
              // if (inv_step_size > 0.0) {
              //   cslip_state = cslip_state - eps_t * grad_cslip2 / inv_step_size;
              // }

              // // clip the squared slip length >= 0
              // if (cslip_state > 0.0) {
              //   cdel_w[ibf] = sqrt(cslip_state) * delta;
              // } else {
              //   cdel_w[ibf] = 0.0;
              // }

              // if (cdel_w[ibf] > 1.5*delta) {
              //   cdel_w[ibf] = 1.5*delta;
              // }

              // recompute the slip state ...

              T_s[ibf] = (T[icv]*cdel_w[ibf] + T_bc*delta)/(cdel_w[ibf]+delta);
              const double rho_bc = p[icv] / (R_gas * T_bc); // actual rho at the wall, not slip
              FOR_I3 {
                rhou_s[ibf][i] = (rho[icv]*u[icv][i]*cdel_w[ibf] + rho_bc*u_wall[ibf][i]*delta)/(cdel_w[ibf]+delta);
              }

              // compute unit_n
              double unit_n[3];
              const double mag_n = MAG(zone_ptr->n_bf[ibf]);
              FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i]/mag_n;

              // compute the mean transpiration ...
              double rhoun_s = 0.0;
              FOR_I3 {
                rhoun_s += rhou_s[ibf][i] * unit_n[i];
              }

              if (t_avg < 0.0) {
                mean_rhoun_s[ibf] = rhoun_s; // we will reduce from here
              } else {
                mean_rhoun_s[ibf] *= (1.0 - eps_t);
                mean_rhoun_s[ibf] += eps_t * rhoun_s;
              }
            }
          }

          if (t_avg < 0.0) {
            // reduce the mean transpiration over all wall faces in this bf zone
            double my_buf[2] = {0.0,0.0};
            double buf[2] = {0.0,0.0};
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              my_buf[0] += zone_ptr->area_bf[ibf];
              my_buf[1] += mean_rhoun_s[ibf] * zone_ptr->area_bf[ibf];
            }
            MPI_Allreduce(my_buf,buf,2,MPI_DOUBLE,MPI_SUM,mpi_comm);
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_rhoun_s[ibf] = buf[1]/buf[0];
            }
          }

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            if (zone_ptr->area_over_delta_bf[ibf] > 0.0) {
              
              const int icv      = zone_ptr->cvobf[ibf];
              const double delta = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf];

              // compute unit_n
              double unit_n[3];
              const double mag_n = MAG(zone_ptr->n_bf[ibf]);
              FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i]/mag_n;

              // correct the slip state ...

              FOR_I3 rhou_s[ibf][i] -= mean_rhoun_s[ibf] * unit_n[i];

              // recompute sgs slip parameters

              const double rho_bc = p[icv] / (R_gas * T_bc); // actual rho at the wall, not slip
              
              if (b_sgs_slip) {
                double dudx_w[3][3];
                FOR_I3 {
                  FOR_J3 {
                    dudx_w[i][j] = (rhou_s[ibf][i]/rho_bc - u[icv][i])/delta*unit_n[j];
                  }
                }
                double dudx_dot_n[3] = {0.0,0.0,0.0};
                FOR_I3 {
                  FOR_J3 {
                    dudx_dot_n[i] += dudx[icv][i][j] * unit_n[j];
                  }
                }
                FOR_I3 {
                  FOR_J3 {
                    dudx_w[i][j] += dudx[icv][i][j] - dudx_dot_n[i]*unit_n[j];
                  }
                }

                for (int i = 0; i < 3; ++i) {
                  sij_w[ibf][i] = dudx_w[i][i];
                  sij_w[ibf][i+3] = 0.5*(dudx_w[i][(i+1)%3] + dudx_w[(i+1)%3][i]);
                }

                double sij_cv[6];
                for (int i = 0; i < 3; ++i) {
                  sij_cv[i] = dudx[icv][i][i];
                  sij_cv[i+3] = 0.5*(dudx[icv][i][(i+1)%3] + dudx[icv][(i+1)%3][i]);
                }

                double dsdn_w[6];
                for (int i = 0; i < 6; ++i) {
                  dsdn_w[i] = (sij_cv[i] - sij_w[ibf][i])/delta;
                }
                
                for (int i = 0; i < 6; ++i) {
                  betaij_w[ibf][i] = sij_w[ibf][i] - cdel_w[ibf]*dsdn_w[i];
                }

                double rij_w[6];
                for (int i = 0; i < 3; ++i) {
                  rij_w[i] = rhou_s[ibf][i]*rhou_s[ibf][i]/rho_bc/rho_bc;
                  rij_w[i+3] = rhou_s[ibf][i]*rhou_s[ibf][(i+1)%3]/rho_bc/rho_bc;
                }

                double SB = 0;
                double BB = 0;
                double RB = 0;
                for (int i = 0; i < 6; ++i) {
                  BB += betaij_w[ibf][i]*betaij_w[ibf][i];
                  SB += betaij_w[ibf][i]*sij_w[ibf][i];
                  RB += betaij_w[ibf][i]*rij_w[i];
                }

                assert(RB == RB);
                assert(SB == SB);
                assert(BB == BB);
                sgs_cdel_w[ibf] = cdel_w[ibf] * SB / (BB + 1e-12);
                sgs_f_w[ibf] = -0.5 * RB / (BB + 1e-12);

                assert( sgs_cdel_w[ibf] >= 0.0);
              }

              // now the flux...

              const double rhou_s_dot_n = DOT_PRODUCT(rhou_s[ibf],zone_ptr->n_bf[ibf]);
              const double rho_wall     = p[icv] / (R_gas * T_s[ibf]);

              double Frhou[3] = { 0.0, 0.0, 0.0 };

              // save the transpiration direction for diagnostics
              if (rhou_s_dot_n > 0.0) {
                mean_direc[ibf] = (1.0-eps_t)*mean_direc[ibf] - eps_t*1.0; // - out
              } else {
                mean_direc[ibf] = (1.0-eps_t)*mean_direc[ibf] + eps_t*1.0; // + in
              }

              if (rhou_s_dot_n > 0.0) {
              
                const double H0 = gogm1*p[icv]/rho[icv] +
                  0.5*DOT_PRODUCT(u[icv],u[icv]);

                rhs[icv][0] -= rhou_s_dot_n;
                FOR_I3 {
                  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
                  rhs[icv][1+i] -= rhou_s_dot_n*u[icv][i];
                  Frhou[i]      -= rhou_s_dot_n*u[icv][i];
                }
                rhs[icv][4] -= rhou_s_dot_n*H0;
              
              } else {
              
                const double H_wall = gogm1 * p[icv] / rho_wall +
                  0.5*DOT_PRODUCT(rhou_s[ibf],rhou_s[ibf])/(rho_wall*rho_wall);

                rhs[icv][0] -= rhou_s_dot_n;
                FOR_I3 {
                  rhs[icv][1+i] -= p[icv]*zone_ptr->n_bf[ibf][i];
                  rhs[icv][1+i] -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                  Frhou[i]      -= rhou_s_dot_n*rhou_s[ibf][i]/rho_wall;
                }
                rhs[icv][4] -= rhou_s_dot_n*H_wall;

              }

              // now the viscous contributions ...

              double visc_coeff = mu_lam[icv] * area_over_delta_bf[ibf];
              const double l_coeff    = loc_lam[icv] * area_over_delta_bf[ibf];

              // now the sgs contributions ...

              if (b_sgs_slip) {
                const double mu_sgs_w = (delta*sgs_f_w[ibf]+sgs_cdel_w[ibf]*mu_sgs[icv])/(delta+sgs_cdel_w[ibf]);
                visc_coeff += mu_sgs_w * area_over_delta_bf[ibf];
              }

              for (int i = 0; i < 3; ++i) {
              
                double tmp = visc_coeff*(rhou_s[ibf][i]/rho_wall - u[icv][i]);
                rhs[icv][i+1] += tmp;
                Frhou[i]      += tmp;
                rhs[icv][4]   += tmp * rhou_s[ibf][i] / rho_wall;
              
              }

              rhs[icv][4] += l_coeff*(R_gas*gogm1*T_s[ibf] - gogm1*p[icv]/rho[icv]);

              // compute the wall stress and store for later
              for (int i = 0; i < 3; ++i)
                tau_wall_[ibf][i] = -Frhou[i] / zone_ptr->area_bf[ibf];

              // compute errors

              // .. we need the direction
              FOR_I3 mean_u1[ibf][i] = (1.0-eps_t)*mean_u1[ibf][i] + eps_t*u[icv][i];
              
              // double Udotn = 0.0;
              // FOR_I3 Udotn += mean_u1[ibf][i] * unit_n[i];
              // double dirU[3];
              // FOR_I3 dirU[i] = mean_u1[ibf][i] - Udotn * unit_n[i];
              // double magU = MAG(dirU);
              // FOR_I3 dirU[i] /= magU;

              double dirU[3] = {1.0,0.0,0.0}; // XXX align with x direction
              double Udotn = 0.0;
              FOR_I3 Udotn += dirU[i] * unit_n[i];
              FOR_I3 dirU[i] -= Udotn * unit_n[i];
              double magU = MAG(dirU);
              FOR_I3 dirU[i] /= magU;

              double tau_wall_t = 0.0;
              FOR_I3 tau_wall_t += dirU[i] * tau_wall_[ibf][i];

              // double tau_wall_n = 0.0;
              // double tau_wall_p[3];
              // double tau_wall_t = 0.0;
              // FOR_I3 tau_wall_n += tau_wall_[ibf][i] * unit_n[i];
              // FOR_I3 tau_wall_p[i] = tau_wall_[ibf][i] - tau_wall_n * unit_n[i];
              // tau_wall_p[2] = 0.0; // XXX ignore spanwise component
              // tau_wall_t = MAG(tau_wall_p);
              // double sign_tau_wall_t = 1.0;
              // if (tau_wall_p[0] < 0.0) {
              //   sign_tau_wall_t = -1.0;
              // }
              // tau_wall_t *= sign_tau_wall_t;

              double cf_current = 2.0*tau_wall_t/rho[icv]/U_inf/U_inf;

              // store running average error
              double this_error = cf_current - cf_ref[ibf];
              double w0 = error_counts[ibf] / (error_counts[ibf] + dt);
              double w1 = dt / (error_counts[ibf] + dt);
              error_avg[ibf] = w0 * error_avg[ibf] + w1 * this_error;
              error_counts[ibf] += dt;

              // // store exponential average error
              // double this_error = cf_current - cf_ref[ibf];
              // const double tau_c = 0.5*dt_optim_interval;
              // double w0 = tau_c/(tau_c+dt);
              // double w1 = dt/(tau_c+dt);
              // error_avg[ibf] = w0 * error_avg[ibf] + w1 * this_error;
              // error_counts[ibf] += dt;

              // store running average cf
              w0 = cf_counts[ibf] / (cf_counts[ibf] + dt);
              w1 = dt / (cf_counts[ibf] + dt);
              cf_avg[ibf] = w0 * cf_avg[ibf] + w1 * cf_current;
              cf_counts[ibf] += dt;

              // compute additional averages
              
              double dudn_bf[3];
              double dupdn_bf[3];
              for (int i = 0; i < 3; ++i) {
                  dudn_bf[i] = (u[icv][i] - rhou_s[ibf][i]/rho[icv])/delta;
              }
              double dundn_bf = 0.0;
              for (int i = 0; i < 3; ++i) {
                dundn_bf += -dudn_bf[i] * unit_n[i];
              }
              for (int i = 0; i < 3; ++i) {
                dupdn_bf[i] = dudn_bf[i] + dundn_bf * unit_n[i];
              }
              double dutdn_bf = 0.0;
              dupdn_bf[2] = 0.0; // XXX discard the spanwise component
              for (int i = 0; i < 3; ++i) {
                dutdn_bf += dupdn_bf[i] * dupdn_bf[i];
              }
              if (dutdn_bf > 0.0) {
                dutdn_bf = sqrt(dutdn_bf);
              } else {
                dutdn_bf = 0.0;
              }
              if (u[icv][0] < 0.0) { // XXX align with x direction
                dutdn_bf *= -1.0;
              }

              double up_cv[3];
              double ut_cv = 0.0;
              double un_cv = 0.0;
              FOR_I3 un_cv += -u[icv][i] * unit_n[i];
              FOR_I3 up_cv[i] = u[icv][i] + un_cv * unit_n[i];
              up_cv[2] = 0.0; // XXX discard the spanwise component
              FOR_I3 ut_cv += up_cv[i] * up_cv[i];
              if (ut_cv > 0.0) {
                ut_cv = sqrt(ut_cv);
              } else {
                ut_cv = 0.0;
              }
              if (u[icv][0] < 0.0) { // XXX align with x direction
                ut_cv *= -1.0;
              }
              
              GLtn_avg[ibf] *= (1.0-eps_t);
              GLtn_avg[ibf] += eps_t * dutdn_bf * dundn_bf;

              DLtn_avg[ibf] *= (1.0-eps_t);
              DLtn_avg[ibf] += eps_t * ut_cv * dundn_bf;

              Stn_avg[ibf] *= (1.0-eps_t);
              Stn_avg[ibf] += eps_t * 0.5 * dutdn_bf;
            
            } else {
            
              for (int i = 0; i < 3; ++i)
                tau_wall_[ibf][i] = 0.0;
            
            }
          
          }

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {}

        void preLoadBalance() {
        
          delete[] cdel_w;    cdel_w    = NULL;
          delete[] rhou_s;    rhou_s    = NULL;
          delete[] T_s;       T_s       = NULL;
          delete[] u_wall;    u_wall    = NULL;
          delete[] tau_wall_; tau_wall_ = NULL;

          if (b_sgs_slip) {
            delete[] sgs_cdel_w; sgs_cdel_w = NULL;
            delete[] sgs_f_w;    sgs_f_w    = NULL;
            delete[] sij_w;      sij_w      = NULL;
            delete[] betaij_w;   betaij_w   = NULL;
          }

          delete[] mean_rhoun_s; mean_rhoun_s = NULL;
          delete[] mean_direc; mean_direc = NULL;
          delete[] mean_u1; mean_u1 = NULL;
          delete[] GLtn_avg; GLtn_avg = NULL;
          delete[] DLtn_avg; DLtn_avg = NULL;
          delete[] Stn_avg; Stn_avg = NULL;

          if (b_modelgrad) {
            delete[] gradJmodel; gradJmodel = NULL;
            delete[] hessJmodel; hessJmodel = NULL;
          }

          delete[] error_avg; error_avg = NULL;
          delete[] error_counts; error_counts = NULL;
          delete[] cf_avg; cf_avg = NULL;
          delete[] cf_counts; cf_counts = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::freeOnDevice(cdel_w_d); cdel_w_d = NULL;
          cti::utils::freeOnDevice(rhou_s_d); rhou_s_d = NULL;
          cti::utils::freeOnDevice(T_s_d);    T_s_d    = NULL;
          cti::utils::freeOnDevice(u_wall_d); u_wall_d = NULL;
          cti::utils::freeOnDevice(tau_wall_d_); tau_wall_d_ = NULL;

          if (b_sgs_slip) {
            cti::utils::freeOnDevice(sgs_cdel_w_d); sgs_cdel_w_d = NULL;
            cti::utils::freeOnDevice(sgs_f_w_d);    sgs_f_w_d    = NULL;
            cti::utils::freeOnDevice(sij_w_d);      sij_w_d      = NULL;
            cti::utils::freeOnDevice(betaij_w_d);   betaij_w_d   = NULL;
          }

          cti::utils::freeOnDevice(mean_rhoun_s_d); mean_rhoun_s_d = NULL;
          cti::utils::freeOnDevice(mean_direc_d); mean_direc_d = NULL;
          cti::utils::freeOnDevice(mean_u1_d); mean_u1_d = NULL;
          cti::utils::freeOnDevice(GLtn_avg_d); GLtn_avg_d = NULL;
          cti::utils::freeOnDevice(DLtn_avg_d); DLtn_avg_d = NULL;
          cti::utils::freeOnDevice(Stn_avg_d); Stn_avg_d = NULL;

          if (b_modelgrad) {
            cti::utils::freeOnDevice(gradJmodel_d); gradJmodel_d = NULL;
            cti::utils::freeOnDevice(hessJmodel_d); hessJmodel_d = NULL;
          }

          cti::utils::freeOnDevice(error_avg_d); error_avg_d = NULL;
          cti::utils::freeOnDevice(error_counts_d); error_counts_d = NULL;
          cti::utils::freeOnDevice(cf_avg_d); cf_avg_d = NULL;
          cti::utils::freeOnDevice(cf_counts_d); cf_counts_d = NULL;

#endif
        
        }

        void postLoadBalance() {
          initData();
        }

        void addBoundaryFluxGpu(cti_ffp * rhs, const int rk_stage, gpuStream_t& stream) {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          gradientUpdateSlipLength(); // perform gradient update at time intervals
          
          const double t_c = min(abs(t_avg),time-time_init);
          const double eps_t = dt/(dt+t_c);

          if ( zone_ptr->nbf > 0 ) {
          
            {
            
              const int block_size = 128;
              const int grid_size = (zone_ptr->nbf + block_size - 1)/block_size;

              LAUNCH_GPU_KERNEL_STREAM( computeOptSlipStateAndTransp, grid_size, block_size,
                                        stream, sp_vol_f_d, u_f_d, p_f_d, h_f_d, mu_lam_d,
                                        loc_lam_d,
                                        (cti_ffp)T_bc, u_wall_d,
                                        cdel_w_d,rhou_s_d,T_s_d,
                                        mean_rhoun_s_d,
                                        tau_wall_d_,
                                        c_smag_d,
                                        zone_ptr->cvobf_d,
                                        zone_ptr->n_bf_d,
                                        zone_ptr->area_bf_d,
                                        zone_ptr->area_over_delta_bf_d,
                                        zone_ptr->nbf,
                                        (cti_ffp)gamma,
                                        (cti_ffp)eps_t,
                                        (cti_ffp)t_avg,
                                        (cti_ffp)c_slip,
                                        (cti_ffp)R_gas);
            
            }

          }

          if (t_avg < 0.0) {
          
            {
            
              const int block_size = 128;
              const int grid_size = 128;

              LAUNCH_GPU_KERNEL( (computeSlipMeanTransp<cti_ffp,block_size>),
                                  grid_size,block_size,
                                  res_d,
                                  mean_rhoun_s_d,
                                  zone_ptr->area_bf_d,
                                  zone_ptr->nbf);
            
            }

            {
            
              // reduce the following in place
              const int block_size = 128; // must match the grid size above...
              const int grid_size = 1;

              cti::utils::reduce(block_size,block_size,grid_size,res_d,res_d);
            
            }

            double my_buf, buf;
            cti::utils::copyToHost(&my_buf,res_d,1);

            MPI_Allreduce(&my_buf,&buf,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

            buf /= zone_ptr->area_global; // area average the mean transpiration
            
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              mean_rhoun_s[ibf] = buf;
            }

            cti::utils::copyToDevice( mean_rhoun_s_d,mean_rhoun_s,zone_ptr->nbf);

          }

          if (zone_ptr->nbf > 0) {
          
            {

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1)/block_size;

              LAUNCH_GPU_KERNEL_STREAM( addWmOptSlipIsoFlux, grid_size, block_size,
                                        stream, rhs,
                                        sp_vol_f_d,
                                        u_f_d,
                                        p_f_d,
                                        h_f_d,
                                        mu_lam_d,
                                        loc_lam_d,
                                        dudx_d,
                                        (cti_ffp)T_bc,
                                        u_wall_d,
                                        cdel_w_d,
                                        rhou_s_d,
                                        T_s_d,
                                        mu_sgs_d,
                                        sgs_cdel_w_d,
                                        sgs_f_w_d,
                                        sij_w_d,
                                        betaij_w_d,
                                        mean_rhoun_s_d,
                                        mean_direc_d,
                                        mean_u1_d,
                                        tau_wall_d_,
                                        GLtn_avg_d,
                                        DLtn_avg_d,
                                        Stn_avg_d,
                                        gradJmodel_d,
                                        hessJmodel_d,
                                        error_avg_d,
                                        error_counts_d,
                                        cf_avg_d,
                                        cf_counts_d,
                                        cf_ref_d,
                                        zone_ptr->cvobf_d,
                                        zone_ptr->n_bf_d,
                                        zone_ptr->area_bf_d,
                                        zone_ptr->area_over_delta_bf_d,
                                        zone_ptr->nbf,
                                        (cti_ffp)gamma,
                                        (cti_ffp)eps_t,
                                        (cti_ffp)U_inf,
                                        (cti_ffp)R_gas,
                                        dt,
                                        dt_optim_interval,
                                        b_sgs_slip,
                                        b_modelgrad);

            }

          }

#endif
        
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}

        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost((double*)rhou_s,rhou_s_d,3*zone_ptr->nbf); 
          cti::utils::copyToHost(cdel_w, cdel_w_d, zone_ptr->nbf); 
          cti::utils::copyToHost(T_s,T_s_d, zone_ptr->nbf); 

          if (b_sgs_slip) {
            cti::utils::copyToHost(sgs_cdel_w, sgs_cdel_w_d, zone_ptr->nbf);
            cti::utils::copyToHost(sgs_f_w, sgs_f_w_d, zone_ptr->nbf);
            cti::utils::copyToHost((double*)sij_w,sij_w_d,6*zone_ptr->nbf);
            cti::utils::copyToHost((double*)betaij_w,betaij_w_d,6*zone_ptr->nbf);
          }

          cti::utils::copyToHost(mean_rhoun_s,mean_rhoun_s_d, zone_ptr->nbf);
          cti::utils::copyToHost(mean_direc,mean_direc_d, zone_ptr->nbf);
          cti::utils::copyToHost((double*)mean_u1,mean_u1_d, 3*zone_ptr->nbf);

          cti::utils::copyToHost(GLtn_avg, GLtn_avg_d, zone_ptr->nbf);
          cti::utils::copyToHost(DLtn_avg, DLtn_avg_d, zone_ptr->nbf);
          cti::utils::copyToHost(Stn_avg, Stn_avg_d, zone_ptr->nbf);

          if (b_modelgrad) {
            cti::utils::copyToHost(gradJmodel, gradJmodel_d, zone_ptr->nbf);
            cti::utils::copyToHost(hessJmodel, hessJmodel_d, zone_ptr->nbf);
          }

          cti::utils::copyToHost((double*)tau_wall_, tau_wall_d_, 3*zone_ptr->nbf);

#endif

          if (b_blde_params) {
            blde->computeBlFromPt(bl_delta,bl_delta_star,bl_theta,bl_u_edge,bl_thresh);
          }
        
        }

        void query() {
       
          // XXX
          double my_buf[3] = {0.0,0.0,0.0};

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          
            double unit_n[3];
            FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i] / MAG(zone_ptr->n_bf[ibf]);

            double rhoun = 0.0;
            FOR_I3 rhoun += unit_n[i] * rhou_s[ibf][i];

            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * rhoun;
            my_buf[2]    += zone_ptr->area_bf[ibf] * mean_rhoun_s[ibf];

          }

          double buf[3] = {0.0,0.0,0.0};
          MPI_Reduce(my_buf, buf, 3, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) {
          
            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step
                 << " 6:time " << time
                 << " 8:mean(rhoun_s) " << buf[1]/buf[0]
                 << " 10:mean_rhoun_s " << buf[2]/buf[0]
                 << endl;
          
          }
        
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) {
        
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            for (int i = 0; i < 3; ++i)
              rval_dn3[ibf][i] = 0.0;

          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::PRESSURE_FORCE) ) {
          
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }

          if ( (f_type==ForceType::TOTAL_FORCE) || (f_type==ForceType::VISCOUS_FORCE) ) {
          
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 rval_dn3[ibf][i] += tau_wall_[ibf][i]*zone_ptr->area_bf[ibf];
            }
          }
        
        }

      };

      class CbcBc : public IdealGasBc {
      public:
	
        // The cbc bcs have to set these vars...
        // only u needs to vary spatially so it can relect the local normal in some cases
	
        double rho_bc;
	double (*u_bc)[3]; 
	double p_bc;
	double h_bc;
	
        cti_ffp *u_bc_d; 

        CbcBc(BfZone* zone_ptr) : IdealGasBc(zone_ptr) {
	  
          rho_bc = BIG_DOUBLE;
	  u_bc   = NULL;
	  p_bc   = BIG_DOUBLE;
	  h_bc   = BIG_DOUBLE;

          u_bc_d = NULL;
	
        }
	
        virtual ~CbcBc() {
	  DELETE(u_bc);
	}
	
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            double flux[5];
	    const int icv = zone_ptr->cvobf[ibf];
	    calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
			    rho[icv],u[icv],p[icv],h[icv],gamma,
			    rho_bc,u_bc[ibf],p_bc,h_bc,gamma);
	    //  mass flux needs to be stored for potential passive scalars..
	    //mf[ibf] = flux[0];
	    FOR_I5 rhs[icv][i] -= flux[i];
	  }
	
        }

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addCbcIgBcFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,  
                                      (cti_ffp)rho_bc, u_bc_d, (cti_ffp) p_bc, 
                                      (cti_ffp)h_bc, zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)gamma) ; 


          } 
        

#endif

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {}
        void rkStepGpu(const double * rk_wgt, const int rk_stage) {}
        void syncToHost() {}

        void query() { 

          // grab the static pressure, temperature and the inlet mass flux ...

          double my_buf[4];

          for (int i = 0; i < 4; ++i) 
            my_buf[i] = 0.0;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const int icv = zone_ptr->cvobf[ibf];
            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * p[icv];
            my_buf[2]    += zone_ptr->area_bf[ibf] * T[icv];
            my_buf[3]    += rho[icv] * DOT_PRODUCT( u[icv], zone_ptr->n_bf[ibf]) ;

          } 

          double buf[4];
          MPI_Reduce(my_buf, buf, 4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) { 

            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:p " << buf[1]/buf[0] 
                 << " 10:T " << buf[2]/buf[0] 
                 << " 12:mdot " << buf[3] 
                 << endl;

          }

        } 


        void force(double (*rval_dn3)[3], const ForceType f_type) { 

          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
  
          if (f_type==ForceType::TOTAL_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc,h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc,h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }
      
      };

      class CbcRupBc : public CbcBc {
      public:
	
        double constant_u_bc[3];
        bool b_mrf_transform;
	
        CbcRupBc(BfZone* zone_ptr,Param * param) : CbcBc(zone_ptr) {
	  
          if (mpi_rank == 0) cout << "CbcRupBc: " << param->str() << endl;
          b_mrf_transform = false;
	  zone_ptr->lb_cost = 50;
	  rho_bc = param->getDouble(1);
	  FOR_I3 constant_u_bc[i] = param->getDouble(2+i);
	  p_bc = param->getDouble(5);

          if ( (param->size() == 7) && (param->getString(6) == "STATIONARY_FRAME")) { 

            if ( mpi_rank == 0 ) 
              cout << " > zone : " << zone_ptr->getName() << "  has stationary frame " << endl;
            
            b_mrf_transform = true;

          } 
	}
	
        void initData() {

          h_bc = gamma*p_bc/((gamma-1.0)*rho_bc); // ideal gas
	  assert(u_bc == NULL);
	  u_bc = new double[zone_ptr->nbf][3];
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    FOR_I3 u_bc[ibf][i] = constant_u_bc[i];
	  }

          if ( b_mrf_transform ) {

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const double r[3] = DIFF(zone_ptr->x_bf[ibf],&frame_rotation[3]);

              u_bc[ibf][0] = u_bc[ibf][0] - frame_rotation[1]*r[2] 
                                          + frame_rotation[2]*r[1];

	      u_bc[ibf][1] = u_bc[ibf][1] - frame_rotation[2]*r[0] 
                                          + frame_rotation[0]*r[2];

	      u_bc[ibf][2] = u_bc[ibf][2] - frame_rotation[0]*r[1] 
                                          + frame_rotation[1]*r[0];

            }

          } 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
#endif

	}
	
        void preLoadBalance() {

          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
#endif

	}

        void postLoadBalance() {
	  initData();
	}
        
      };

      class CbcTotalPtBc : public CbcBc { 
      public:

        double un_bc;
        double total_p;
        double total_t;
        double t_relax;

	bool b_frame_rotation;
	double (*u_fr)[3]; // the velocity field due to frame rotation ..

        cti_ffp * u_fr_d;

        double* res_d;

        CbcTotalPtBc(BfZone* zone_ptr, Param * param) : CbcBc(zone_ptr) { 

          un_bc = 0.0;
          registerData(un_bc,zone_ptr->name+":un_bc",READ_DATA|WRITE_DATA);

	  b_frame_rotation = false;

          int ierr = 0;
          int iarg = 1;
          while ( iarg < param->size()) { 

            string token = param->getUCString(iarg++);
            if ( token == "P_TOTAL") { 

              total_p = param->getDouble(iarg++);
              ierr   |= (1<<0);

            } 
            else if ( token == "T_TOTAL") { 

              total_t = param->getDouble(iarg++);
              ierr   |= (1<<1);

            } 
            else if ( token == "T_RELAX") { 

              t_relax = param->getDouble(iarg++);
              ierr  |= (1<<2);

            }
	    else if ( token == "STATIONARY_FRAME") {
	      if ( frame_rotation != NULL) {
		b_frame_rotation = true;
	      }
	    }
            else { 
              CERR( " > unrecognized CBC_TOTAL_PT token: " << token); 
            }

          } 

          if ( (ierr < 7) || (ierr > 8) ) { 
            CERR( " > CBC_TOTAL_PT requires params: P_TOTAL <double> \
                T_TOTAL <double> T_RELAX <double> \
                or: P_TOTAL <double>  T_TOTAL <double> \
                T_RELAX <double> STATIONARY_FRAME");
          } 

	  u_fr   = NULL;
          u_fr_d = NULL;

          res_d = NULL;

        }

	~CbcTotalPtBc() {
	  DELETE(u_fr);
	}

        void computeRhoHPFromTotalConditions(double& rho_, double& h_, double& p_) { 

          const double total_rh = total_p/R_gas/total_t;
          const double cp       = R_gas * gamma/(gamma-1.0);
          const double T_bc     = total_t - 0.5*un_bc*un_bc/cp;
          rho_                  = total_rh* pow(T_bc/total_t, 1.0/(gamma-1.0));
          p_                    = total_p * pow(T_bc/total_t, gamma/(gamma-1.0));
          h_                    = cp * T_bc;

        } 

        void initData() { 

          computeRhoHPFromTotalConditions(rho_bc,h_bc,p_bc);

	  assert( u_fr == NULL);
	  u_fr = new double[zone_ptr->nbf][3];
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
	    for (int i = 0; i < 3; ++i)
	      u_fr[ibf][i] = 0.0;

	  if (b_frame_rotation) {
            double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_fr[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];

	      u_fr[ibf][1] = -frame_rotation[2]*r[0] +
                              frame_rotation[0]*r[2];

	      u_fr[ibf][2] = -frame_rotation[0]*r[1] +
                              frame_rotation[1]*r[0];
	    }
	  }

          assert( u_bc == NULL);
          u_bc = new double[zone_ptr->nbf][3];
          
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

	    const double mag_n = MAG(zone_ptr->n_bf[ibf]);
	    double unit_n[3];
	    FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

	    double ut[3];
	    double ufr_n = DOT_PRODUCT(u_fr[ibf],unit_n);
	    for (int i = 0; i < 3; ++i)
	      ut[i] = u_fr[ibf][i] - ufr_n*unit_n[i];

	    for (int i = 0; i < 3; ++i) 
	      u_bc[ibf][i] = un_bc*unit_n[i] + ut[i];

          } 
         
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::initAndCopyToDevice(u_bc_d,(double*) u_bc, 3*zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_fr_d,(double*) u_fr, 3*zone_ptr->nbf);
          cti::utils::initOnDevice(res_d,1024);
#endif

        } 

        void preLoadBalance() { 

          assert( u_bc); 
          delete[] u_bc; u_bc= NULL;

          assert( u_fr);
          delete[] u_fr; u_fr= NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
          cti::utils::freeOnDevice(res_d);  res_d  = NULL;
          cti::utils::freeOnDevice(u_fr_d); u_fr_d = NULL;
#endif

        } 

        void postLoadBalance() { 
          initData();
        }


        // use the rk step in order to advance the value of the un_bc at the boundary... 

        void rkStep(const double *rk_wgt, const int rk_stage) { 

          if ( rk_stage == 1)  { 

            const double eps   = 1.0 / (t_relax/dt + 1.0);

            double my_buf = 0.0;
            double buf;
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const int icv = zone_ptr->cvobf[ibf];
              my_buf += DOT_PRODUCT(u[icv],zone_ptr->n_bf[ibf]);


            } 

            MPI_Allreduce(&my_buf,&buf,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

            buf /= zone_ptr->area_global; // face average un... 
            un_bc = (1.0-eps)*un_bc + eps*buf;
            computeRhoHPFromTotalConditions(rho_bc,h_bc,p_bc);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const double mag_n = MAG(zone_ptr->n_bf[ibf]);
	      double unit_n[3];
	      FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

	      double ut[3];
	      double ufr_n = DOT_PRODUCT(u_fr[ibf],unit_n);
	      for (int i = 0; i < 3; ++i)
		ut[i] = u_fr[ibf][i] - ufr_n*unit_n[i];

              for (int i = 0; i < 3; ++i) 
                u_bc[ibf][i] = un_bc*unit_n[i] + ut[i];

            }

          }
        }


        void rkStepGpu(const double * rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( rk_stage == 1) { 

            { 

              const int block_size = 128;
              const int grid_size  = 128; 

              LAUNCH_GPU_KERNEL( (computeUnDp<cti_ffp,block_size>), 
                                 grid_size, block_size, res_d, u_d, 
                                 zone_ptr->cvobf_d, zone_ptr->n_bf_d, zone_ptr->nbf);

            }

            { 

              // reduce the following in place .. 

              const int block_size = 128; // must match the grid size above ... 
              const int grid_size  = 1;

              cti::utils::reduce(block_size,block_size,grid_size,res_d,res_d);

            } 

            double my_buf, buf; 
            cti::utils::copyToHost(&my_buf,res_d,1);
          
            MPI_Allreduce(&my_buf,&buf,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

            buf /= zone_ptr->area_global; // face average un... 
            const double eps   = 1.0 / (t_relax/dt + 1.0);
            un_bc = (1.0-eps)*un_bc + eps*buf;


            computeRhoHPFromTotalConditions(rho_bc,h_bc,p_bc);

            if ( zone_ptr->nbf > 0 ) { 

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

              LAUNCH_GPU_KERNEL( setUbcFromUn, grid_size, block_size, u_bc_d, 
                                 zone_ptr->n_bf_d, un_bc, u_fr_d, zone_ptr->nbf);


            } 

          }
#endif

        }

        void syncToHost() {}

        void query() { 

          // grab the static pressure, temperature and the inlet mass flux ...

          double my_buf[4];
          for (int i = 0; i < 4; ++i) 
            my_buf[i] = 0.0;


          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const int icv = zone_ptr->cvobf[ibf];
            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * p[icv];
            my_buf[2]    += zone_ptr->area_bf[ibf] * T[icv];
            my_buf[3]    += zone_ptr->area_bf[ibf] * rho[icv] * un_bc;

          } 

          double buf[4];
          MPI_Reduce(my_buf, buf, 4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) { 

            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:p " << buf[1]/buf[0] 
                 << " 10:T " << buf[2]/buf[0] 
                 << " 12:mdot " << buf[3] 
                 << endl;

          }

        } 

      };

#define NUM_LIMIT 32
      
      class CbcTotalPtProfileBc : public IdealGasBc {
      public:

        double *total_p;
        double *total_t;

	double *angle_t;
	double *angle_r;

	double t_relax;

	double *rho_bc;
        double *p_bc;
        double *h_bc;
	double (*u_bc)[3];
        double *un_bc;
	
	bool b_swirl;

	int num;
	double *un_bc_prof;
	double *area_in_bin;

	double r_min;
	double r_max;
	double delta_r;
	double *r_bf;

        cti_ffp *total_p_d;
        cti_ffp *total_t_d;
	
	cti_ffp *angle_t_d;
	cti_ffp *angle_r_d;

	cti_ffp *r_bf_d;

	cti_ffp *xc_d;
	cti_ffp *axis_d;
	
        double *rho_bc_d;
        double *u_bc_d;
        double *p_bc_d;
	double *h_bc_d;

	double *un_bc_prof_d;
	double *un_bc_d;
	double *work_arr_d;
	
        string filename;
        string fileformat;
        double xc[3],axis[3];
	
        string helpString() {
          return "\n  Examples:\n<zonename> CBC_TOTAL_PT_PROFILE FILE <filename> FORMAT FLUENT_PROFILE T_RELAX <t_relax>"
                              "\n<zonename> CBC_TOTAL_PT_PROFILE FILE <filename> FORMAT ASCII T_RELAX <t_relax>"
                              "\n<zonename> CBC_TOTAL_PT_PROFILE FILE <filename> FORMAT ASCII XC <x0> <y0> <z0> AXIS <n0> <n1> <n2> T_RELAX <t_relax>"
	                      "\n<zonename> CBC_TOTAL_PT_PROFILE FILE <filename> FORMAT ASCII XC <x0> <y0> <z0> AXIS <n0> <n1> <n2> SWIRL T_RELAX <t_relax>"
	                      "\n<zonename> CBC_TOTAL_PT_PROFILE FILE <filename> FORMAT ASCII XC <x0> <y0> <z0> AXIS <n0> <n1> <n2> SWIRL T_RELAX <t_relax> N <num>";
        }

        CbcTotalPtProfileBc(BfZone* zone_ptr,Param * param) : IdealGasBc(zone_ptr) {
          if (mpi_rank == 0) cout << "CbcTotalPtProfileBc: " << param->str() << endl;

	  b_swirl = false;
	  t_relax = BIG_DOUBLE;

	  total_p = NULL;
	  total_t = NULL;

	  angle_t = NULL;
	  angle_r = NULL;
	  
          un_bc  = NULL; registerData(un_bc ,zone_ptr->nbf,zone_ptr->name+":un_bc",READ_DATA|WRITE_DATA);

	  rho_bc = NULL; registerData(rho_bc,zone_ptr->nbf,zone_ptr->name+":rho_bc");  
          u_bc   = NULL; registerData(u_bc  ,zone_ptr->nbf,zone_ptr->name+":u_bc"  );
          p_bc   = NULL; registerData(p_bc  ,zone_ptr->nbf,zone_ptr->name+":p_bc"  );
	  h_bc   = NULL; registerData(h_bc  ,zone_ptr->nbf,zone_ptr->name+":h_bc"  );

	  num = NUM_LIMIT;
	  un_bc_prof  = NULL;
	  area_in_bin = NULL;

	  r_min = BIG_DOUBLE;
	  r_max = BIG_DOUBLE;
	  delta_r = BIG_DOUBLE;

	  r_bf       = NULL;

	  total_p_d  = NULL;
	  total_t_d  = NULL;

	  angle_t_d  = NULL;
	  angle_r_d  = NULL;
	  
	  r_bf_d     = NULL;
	  
	  xc_d       = NULL;
	  axis_d     = NULL;
	  
          rho_bc_d   = NULL;
          u_bc_d     = NULL;
          p_bc_d     = NULL;
	  h_bc_d     = NULL;

	  un_bc_prof_d = NULL;
	  un_bc_d      = NULL;
	  work_arr_d = NULL;
	  
	  fileformat = "FLUENT_PROFILE";
          FOR_I3 xc[i] = zone_ptr->x_global[i];
          FOR_I3 axis[i] = -zone_ptr->n_global[i]; // point into domain
	  const double mag = sqrt(DOT_PRODUCT(axis,axis));
	  assert ( mag > 0.0) ;
	  FOR_I3 axis[i] /= mag ;   // make sure it is a unit vector

          int iarg = 1;
          while (iarg < param->size()) {
            string token = param->getUCString(iarg++);
            if (token == "FILE") {
              filename = param->getString(iarg++);
              if (mpi_rank == 0) cout << " > file name: " << filename << endl;
            }
            else if (token == "FORMAT") {
              fileformat = param->getUCString(iarg++);
              if (mpi_rank == 0) cout << " > file format: " << fileformat << endl;
            }
            else if (token == "XC") {
              FOR_I3 xc[i] = param->getDouble(iarg++);
            }
            else if (token == "AXIS") {
              FOR_I3 axis[i] = param->getDouble(iarg++);
	      const double mag = sqrt(DOT_PRODUCT(axis,axis));
              assert ( mag > 0.0) ;
              FOR_I3 axis[i] /= mag ;   // make sure it is a unit vector
            }
	    else if ( token == "SWIRL") {
	      b_swirl = true;
            }	    
	    else if ( token == "T_RELAX") { 
              t_relax = param->getDouble(iarg++);
            }
	    else if ( token == "N") {
	      num = param->getInt(iarg++);
	    }
            else {
              CERR("unrecognized token " << token << " during parsing of bc:\n" <<
                   param->str() << helpString());
            }
          }

          if (filename == "") {
            CERR("CBC_TOTAL_PT_PROFILE expects FILE <filename>"); 
          }
          if ((fileformat != "FLUENT_PROFILE")&&(fileformat != "ASCII")) {
            CERR("CBC_TOTAL_PT_PROFILE expects FORMAT {FLUENT_PROFILE,ASCII}, where FLUENT_PROFILE is default"); 
          }
	  if (t_relax == BIG_DOUBLE) {
	    CERR("CBC_TOTAL_PT_PROFILE expects T_RELAX <double>");
	  }
	  if (num <= 0) {
	    CERR("CBC_TOTAL_PT_PROFILE expects a positive N <int>");
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

	  if (num > NUM_LIMIT) {
	    COUT("WARNING: N > " << NUM_LIMIT << " is not supported. Set N to " << NUM_LIMIT);
	    num = NUM_LIMIT;
	  }
          
#endif
	  
        }

        ~CbcTotalPtProfileBc() {

	  DELETE(rho_bc);
          DELETE(u_bc);
          DELETE(p_bc);
          DELETE(h_bc);
	  DELETE(un_bc);
	  
	  DELETE(un_bc_prof);
	  DELETE(area_in_bin);

	  DELETE(r_bf);
	  
	  DELETE(total_p);
	  DELETE(total_t);
	  
	  DELETE(angle_t);
	  DELETE(angle_r);
	  
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if (total_p_d)   cti::utils::freeOnDevice(total_p_d);
          if (total_t_d)   cti::utils::freeOnDevice(total_t_d);
	  
          if (angle_t_d)   cti::utils::freeOnDevice(angle_t_d);
	  if (angle_r_d)   cti::utils::freeOnDevice(angle_r_d);

	  if (r_bf_d)     cti::utils::freeOnDevice(r_bf_d);

	  if (xc_d)       cti::utils::freeOnDevice(xc_d);
          if (axis_d)     cti::utils::freeOnDevice(axis_d);

          if (rho_bc_d)   cti::utils::freeOnDevice(rho_bc_d);
          if (u_bc_d)     cti::utils::freeOnDevice(u_bc_d);
          if (p_bc_d)     cti::utils::freeOnDevice(p_bc_d);

	  if (un_bc_prof_d)  cti::utils::freeOnDevice(un_bc_prof_d);
	  if (un_bc_d)       cti::utils::freeOnDevice(un_bc_d);
	  if (work_arr_d)    cti::utils::freeOnDevice(work_arr_d);
          
#endif
	  
        } 

	// moved to bcs_kernels
	/*
	void computeRhoHPFromTotalConditions(double total_p, double total_t, double u_bc[3], double& rho_, double& h_, double& p_) { 

          const double total_rh = total_p/R_gas/total_t;
          const double cp       = R_gas * gamma/(gamma-1.0);
          const double T_bc     = total_t - 0.5*(u_bc[0]*u_bc[0] + u_bc[1]*u_bc[1] + u_bc[2]*u_bc[2])/cp;
          rho_                  = total_rh* pow(T_bc/total_t, 1.0/(gamma-1.0));
          p_                    = total_p * pow(T_bc/total_t, gamma/(gamma-1.0));
          h_                    = cp * T_bc;

        }
	*/

        void initData() {

	  assert(rho_bc);
	  assert(p_bc);
	  assert(h_bc);
	  assert(u_bc);
	  assert(un_bc);

          if ( !checkInit(un_bc,zone_ptr->nbf)) { 

            if ( mpi_rank == 0 ) 
              cout << " > resetting un_bc for zone: " << zone_ptr->getName() << endl;

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      un_bc[ibf] = DOT_PRODUCT( u[icv], zone_ptr->n_bf[ibf]) / MAG(zone_ptr->n_bf[ibf]);
            }
	    
          }
	  
	  assert(total_p == NULL);  total_p = new double[zone_ptr->nbf];
	  assert(total_t == NULL);  total_t = new double[zone_ptr->nbf];
	  
	  assert(angle_t == NULL);  angle_t = new double[zone_ptr->nbf];
	  assert(angle_r == NULL);  angle_r = new double[zone_ptr->nbf];
	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    angle_t[ibf] = 0;
	    angle_r[ibf] = 0;
	  }

	  // find r range of the bf zone
	  double my_minmax[2] = {BIG_DOUBLE, BIG_DOUBLE};
	  for (int ibf = zone_ptr->ibf_begin; ibf != zone_ptr->ibf_end; ++ibf) {
	    for (int nob = noobf_i[ibf]; nob != noobf_i[ibf+1]; ++nob) {
	      const int ino = noobf_v[nob];
	      double dx[3] = DIFF(x_no[ino],xc);
              const double dxn = DOT_PRODUCT(dx,axis);
              FOR_I3 dx[i] -= dxn*axis[i];
              const double r = MAG(dx);
	      my_minmax[0] = min(my_minmax[0], r);
	      my_minmax[1] = min(my_minmax[1], -r);
	    }
	  }
	  double minmax[2];
	  MPI_Allreduce(my_minmax,minmax,2,MPI_DOUBLE,MPI_MIN,mpi_comm);
	  r_min = minmax[0];
	  r_max = -minmax[1];
	  if (mpi_rank == 0) cout << "The r range of bf zone " << zone_ptr->getName() << ": " << r_min << ":" << r_max << endl;

	  
	  bool done = false;
	  double *my_un_prof = NULL;
	  double *buf = NULL;
	  
	  while (!done) {
	    
	    assert(num > 0);
	    delta_r = (r_max-r_min)/double(num);
	    if (mpi_rank == 0) cout << "num = " << num << ", delta_r = " << delta_r << endl;

	    // calc r_bf
	    assert(r_bf == NULL);   r_bf = new double[zone_ptr->nbf];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      double dx[3] = DIFF(zone_ptr->x_bf[ibf],xc);
	      const double dxn = DOT_PRODUCT(dx,axis);
	      FOR_I3 dx[i] -= dxn*axis[i];
	      r_bf[ibf] = MAG(dx);
	    }
	  
	    // circumferential avg for un profile
	    assert(my_un_prof == NULL);   my_un_prof = new double[num*2];
	    for (int i = 0; i < num*2; ++i)
	      my_un_prof[i] = 0.0;

	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int idx = int((r_bf[ibf]-r_min)/delta_r);
	      assert((idx>=0)&&(idx<num));
	      my_un_prof[2*idx]   += un_bc[ibf]*zone_ptr->area_bf[ibf];
	      my_un_prof[2*idx+1] += zone_ptr->area_bf[ibf];
	    }

	    assert(buf == NULL);   buf = new double[num*2];
	    MPI_Allreduce(my_un_prof,buf,2*num,MPI_DOUBLE,MPI_SUM,mpi_comm);

	    bool b_zero_area = false;
	    for (int i = 0; i < num; ++i) {
	      if (buf[2*i+1] == 0.0) {
		b_zero_area = true;
		break;
	      }
	    }
	    if (b_zero_area) {
	      DELETE(r_bf);
	      DELETE(my_un_prof);
	      DELETE(buf);
	      num--;
	      if (mpi_rank == 0) cout << "CBC_TOTAL_PT_PROFILE: zero area in un avg; reduce N to " << num << endl;
	    }
	    else {
	      // store the sum of area_bf in the bins..
	      assert(area_in_bin == NULL);
	      area_in_bin = new double[num];
	      for (int i = 0; i < num; ++i) area_in_bin[i] = buf[2*i+1];
	      done = true;
	    }
	    
	  }

	  
	  assert(un_bc_prof == NULL); un_bc_prof = new double[num];
	  for (int i = 0; i < num; ++i) un_bc_prof[i] = buf[2*i]/buf[2*i+1];
	  
	  // check...
	  /*
	  if (mpi_rank == 0) {
	    cout << " check buf..." << endl;
	    for (int i = 0; i < num; ++i) cout << buf[2*i] << ", " << buf[2*i+1] << endl;
	    cout << " check un_bc_prof... " << endl;
	    for (int i = 0; i < num; ++i) cout << un_bc_prof[i] << endl;
	  }
	  MPI_PAUSE("after checking un_bc_prof"); 
	  */
	  
	  delete[] my_un_prof;
	  delete[] buf;

	  // assign un to each bf
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    const int idx = int((r_bf[ibf]-r_min)/delta_r);
	    assert((idx>=0)&&(idx<num));
	    double grad_un = 0.0;
	    if (idx == 0) {
	      grad_un = (un_bc_prof[idx+1]-un_bc_prof[idx])/delta_r;
	    }
	    else if (idx == num-1) {
	      grad_un = (un_bc_prof[idx]-un_bc_prof[idx-1])/delta_r;
	    }
	    else {
	      grad_un = (un_bc_prof[idx+1] - un_bc_prof[idx-1])/2.0/delta_r;
	    }
	    un_bc[ibf] = un_bc_prof[idx] + grad_un*(r_bf[ibf] - r_min - (double(idx)+0.5)*delta_r);
	  }

	  // read profiles from input file
          utils::ProfileReader profile;
          profile.init(filename,fileformat);

          profile.ensureVar("total-pressure");
          profile.ensureVar("total-temperature");
	  if (b_swirl) {
	    profile.ensureVar("angle-tangential");
	    profile.ensureVar("angle-radial");
	  }
          // profile.printVarsInFile();

          if (profile.getType() == utils::LINE_R) {
            profile.setPoints(r_bf,zone_ptr->nbf);
          }
          else if (profile.getType() == utils::LINE_WD) {
            // need to pass wall-distance for each ibf
            double * wall_dist = new double[zone_ptr->nbf];
            CtiData *dn = eval("distance(\"wall\")",true);
            double *dn_data = dn->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              wall_dist[ibf] = dn_data[zone_ptr->cvobf[ibf]];
            }
            profile.setPoints(wall_dist,zone_ptr->nbf);
            DELETE(wall_dist);
          }
          else {
            assert(profile.getType() == utils::POINT_3D);
            profile.setPoints(zone_ptr->x_bf,zone_ptr->x_global,zone_ptr->nbf);
          }

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    total_p[ibf] = profile.getData(ibf,"total-pressure");
	    total_t[ibf] = profile.getData(ibf,"total-temperature");
	    if (b_swirl) {
	      angle_t[ibf] = profile.getData(ibf,"angle-tangential") * M_PI/180.0;       // deg -> rad
	      angle_r[ibf] = profile.getData(ibf,"angle-radial") * M_PI/180.0;           // deg -> rad
	    }
	    double r[3] = DIFF(zone_ptr->x_bf[ibf],xc);
	    const double dp = DOT_PRODUCT(r,axis);
	    FOR_I3 r[i] -= dp*axis[i]; // r[3]
	    const double r_mag = MAG(r); assert(r_mag > 0.0);
	    const double theta[3] = CROSS_PRODUCT(axis,r);
	    const double theta_mag = MAG(theta); assert(theta_mag > 0.0);
	    const double mag_n = MAG(zone_ptr->n_bf[ibf]);
	    double unit_n[3];
	    FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;
	    // assume n is perpendicular to theta -- is this a reasonable assumption?
	    const double ut = un_bc[ibf]*tan(angle_t[ibf]);
	    const double ur = un_bc[ibf]*tan(angle_r[ibf]);
	    FOR_I3 u_bc[ibf][i] = un_bc[ibf]*unit_n[i]+ut*theta[i]/theta_mag+ur*r[i]/r_mag;
	    computeRhoHPFromTotalConditions(total_p[ibf], total_t[ibf], u_bc[ibf], rho_bc[ibf], h_bc[ibf], p_bc[ibf], gamma, R_gas);
          }


#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          assert(total_p_d == NULL);
          assert(total_t_d == NULL);
	  assert(angle_r_d == NULL);
          assert(angle_t_d == NULL);
          cti::utils::initAndCopyToDevice(total_p_d,total_p,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(total_t_d,total_t,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(angle_r_d,angle_r,zone_ptr->nbf);
	  cti::utils::initAndCopyToDevice(angle_t_d,angle_t,zone_ptr->nbf);

          assert(r_bf_d == NULL);
	  cti::utils::initAndCopyToDevice(r_bf_d,r_bf,zone_ptr->nbf);

	  assert(xc_d == NULL);
	  cti::utils::initAndCopyToDevice(xc_d,(double*)xc,3);

	  assert(axis_d == NULL);
	  cti::utils::initAndCopyToDevice(axis_d,(double*)axis,3);

          assert(rho_bc_d == NULL);
          assert(u_bc_d == NULL);
          assert(p_bc_d == NULL);
          assert(h_bc_d == NULL);
          cti::utils::initAndCopyToDevice(rho_bc_d,rho_bc,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(p_bc_d,p_bc,zone_ptr->nbf);
          cti::utils::initAndCopyToDevice(h_bc_d,h_bc,zone_ptr->nbf);

	  assert(un_bc_prof_d == NULL);
	  cti::utils::initOnDevice(un_bc_prof_d,num);
	  
	  assert(un_bc_d == NULL);
	  cti::utils::initOnDevice(un_bc_d,zone_ptr->nbf);

	  assert(work_arr_d == NULL);
	  cti::utils::initOnDevice(work_arr_d,1024*64);
#endif
        }

        void preLoadBalance() {

          // the boundary condition states are registered, so we dont need 
          // to do anything with those, but we need to take down the device 
          // memory here ... 

	  DELETE(un_bc_prof);
	  DELETE(area_in_bin);

	  DELETE(r_bf);
	  
	  DELETE(total_p);
	  DELETE(total_t);
	  
	  DELETE(angle_t);
	  DELETE(angle_r);

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(total_p_d); total_p_d = NULL;
          cti::utils::freeOnDevice(total_t_d); total_t_d = NULL;
	  
          cti::utils::freeOnDevice(angle_t_d); angle_t_d = NULL;
	  cti::utils::freeOnDevice(angle_r_d); angle_r_d = NULL;

	  cti::utils::freeOnDevice(r_bf_d);   r_bf_d   = NULL;

          cti::utils::freeOnDevice(rho_bc_d); rho_bc_d = NULL;
          cti::utils::freeOnDevice(u_bc_d);   u_bc_d   = NULL;
          cti::utils::freeOnDevice(p_bc_d);   p_bc_d   = NULL;
          cti::utils::freeOnDevice(h_bc_d);   h_bc_d   = NULL;

	  cti::utils::freeOnDevice(xc_d);     xc_d     = NULL;
          cti::utils::freeOnDevice(axis_d);   axis_d   = NULL;

	  cti::utils::freeOnDevice(un_bc_prof_d);  un_bc_prof_d  = NULL;
	  cti::utils::freeOnDevice(un_bc_d);       un_bc_d       = NULL;

	  cti::utils::freeOnDevice(work_arr_d);  work_arr_d  = NULL;
#endif
        }

        void postLoadBalance() {
          initData();
        }

        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::copyToHost(rho_bc,rho_bc_d,zone_ptr->nbf);
          cti::utils::copyToHost((double*)u_bc,u_bc_d,zone_ptr->nbf*3);
          cti::utils::copyToHost(p_bc,p_bc_d,zone_ptr->nbf);
          cti::utils::copyToHost(h_bc,h_bc_d,zone_ptr->nbf);
	  cti::utils::copyToHost(un_bc,un_bc_d,zone_ptr->nbf);

#endif

        } 

        void query() { 

          // grab the static pressure, temperature and the inlet mass flux ...

          double my_buf[4];

          for (int i = 0; i < 4; ++i) 
            my_buf[i] = 0.0;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const int icv = zone_ptr->cvobf[ibf];
            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * p[icv];
            my_buf[2]    += zone_ptr->area_bf[ibf] * T[icv];
            my_buf[3]    += rho[icv] * DOT_PRODUCT( u[icv], zone_ptr->n_bf[ibf]) ;

          } 

          double buf[4];
          MPI_Reduce(my_buf, buf, 4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) { 

            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:p " << buf[1]/buf[0] 
                 << " 10:T " << buf[2]/buf[0] 
                 << " 12:mdot " << buf[3]
                 << endl; 
          }

        }

	void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            double flux[5];
	    const int icv = zone_ptr->cvobf[ibf];
	    calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
			    rho[icv],u[icv],p[icv],h[icv],gamma,
			    rho_bc[ibf],u_bc[ibf],p_bc[ibf],h_bc[ibf],gamma);
	    //  mass flux needs to be stored for potential passive scalars..
	    //mf[ibf] = flux[0];
	    FOR_I5 rhs[icv][i] -= flux[i];
	  }
	}

        void rkStep(const double *rk_wgt,const int rk_stage) {
          if ( rk_stage == 1)  { 

            const double eps   = 1.0 / (t_relax/dt + 1.0);

	    // circumferential un avg
	    double *my_un_prof = new double[num];
	    for (int i = 0; i < num; ++i)
	      my_un_prof[i] = 0.0;

	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	      const int icv = zone_ptr->cvobf[ibf];
	      double unit_n[3];
	      FOR_I3 unit_n[i] = zone_ptr->n_bf[ibf][i] / zone_ptr->area_bf[ibf];
	      const int idx = int((r_bf[ibf]-r_min)/delta_r);
	      assert((idx>=0)&&(idx<num));
	      my_un_prof[idx]   += DOT_PRODUCT(u[icv],unit_n)*zone_ptr->area_bf[ibf];
	    }

	    double *buf = new double[num];
	    MPI_Allreduce(my_un_prof,buf,num,MPI_DOUBLE,MPI_SUM,mpi_comm);
	    
	    for (int i = 0; i < num; ++i)
	      un_bc_prof[i] = (1.0-eps)*un_bc_prof[i] + eps*buf[i]/area_in_bin[i];

	    delete[] my_un_prof;
	    delete[] buf;
	    
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

	      const int idx = int((r_bf[ibf]-r_min)/delta_r);
	      double grad_un = 0.0;
	      if (idx == 0) {
		grad_un = (un_bc_prof[idx+1]-un_bc_prof[idx])/delta_r;
	      }
	      else if (idx == num-1) {
		grad_un = (un_bc_prof[idx]-un_bc_prof[idx-1])/delta_r;
	      }
	      else {
		grad_un = (un_bc_prof[idx+1] - un_bc_prof[idx-1])/2.0/delta_r;
	      }
	      un_bc[ibf] = un_bc_prof[idx] + grad_un*(r_bf[ibf] - r_min - (double(idx)+0.5)*delta_r);

	      double r[3] = DIFF(zone_ptr->x_bf[ibf],xc);
	      const double dp = DOT_PRODUCT(r,axis);
	      FOR_I3 r[i] -= dp*axis[i]; // r[3]
	      const double r_mag = MAG(r); assert(r_mag > 0.0);
	      const double theta[3] = CROSS_PRODUCT(axis,r);
	      const double theta_mag = MAG(theta); assert(theta_mag > 0.0);
	      // assume n is perpendicular to theta -- is this a reasonable assumption?
	      const double sign = (double)sgn(DOT_PRODUCT(axis,zone_ptr->n_bf[ibf]));
	      const double ut = sign*un_bc[ibf]*tan(angle_t[ibf]);
	      const double ur = sign*un_bc[ibf]*tan(angle_r[ibf]);
	      const double n_mag = MAG(zone_ptr->n_bf[ibf]); assert(n_mag > 0.0);
	      FOR_I3 u_bc[ibf][i] = un_bc[ibf]*zone_ptr->n_bf[ibf][i]/n_mag+ut*theta[i]/theta_mag+ur*r[i]/r_mag;
	      computeRhoHPFromTotalConditions(total_p[ibf], total_t[ibf], u_bc[ibf], rho_bc[ibf], h_bc[ibf], p_bc[ibf], gamma, R_gas);
	      
	    }
          }
	}


	void rkStepGpu(const double * rk_wgt, const int rk_stage) {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( rk_stage == 1) { 

            { 

              const int block_size = 256;
              const int grid_size  = 512; // must be smaller than max block size...

              LAUNCH_GPU_KERNEL( reduceUnBlock, 
                                 grid_size, block_size, work_arr_d, u_d, 
                                 zone_ptr->cvobf_d, zone_ptr->n_bf_d,
				 zone_ptr->area_bf_d, r_bf_d, (cti_ffp)r_min, (cti_ffp)delta_r,
				 zone_ptr->nbf);

            }

	    {

              const int block_size = 512; // must equal grid_size above.. 
              const int grid_size  =   1;
              int offset = 0;

	      assert(num <= NUM_LIMIT);
              for (int i = 0; i < num; ++i) {

                // reduce in place... 

                cti::utils::reduce(block_size,block_size,grid_size,
                    work_arr_d+offset,work_arr_d+i); // note the index

                offset += block_size;

              }

            }

            // we need to copy back the value in red_work_arr[0] to the host
            // and then pass through the MPI reduce

	    assert(num <= NUM_LIMIT);
	    double *my_buf = new double[num];
            cti::utils::copyToHost(my_buf,work_arr_d,num);
	    
	    double *buf = new double[num];
            MPI_Allreduce(my_buf,buf,num,MPI_DOUBLE,MPI_SUM,mpi_comm);
	    
            const double eps   = 1.0 / (t_relax/dt + 1.0);
	    for (int i = 0; i < num; ++i)
	      un_bc_prof[i] = (1.0-eps)*un_bc_prof[i] + eps*buf[i]/area_in_bin[i];

            cti::utils::copyToDevice(un_bc_prof_d, un_bc_prof, num);
	    
            if ( zone_ptr->nbf > 0 ) { 

              const int block_size = 128;
              const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

	      LAUNCH_GPU_KERNEL( setUbcFromUnProfile, grid_size, block_size, u_bc_d, rho_bc_d, p_bc_d, h_bc_d, un_bc_d,
				 zone_ptr->n_bf_d, zone_ptr->area_bf_d, zone_ptr->x_bf_d, un_bc_prof_d, total_p_d, total_t_d,
				 angle_t_d, angle_r_d, r_bf_d, xc_d, axis_d, (cti_ffp)r_min, (cti_ffp)delta_r, zone_ptr->nbf,
				 num, (cti_ffp)gamma, (cti_ffp)R_gas);

            }
	    
	    delete[] my_buf;
	    delete[] buf;

          }
#endif
	}

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

	    LAUNCH_GPU_KERNEL_STREAM( addCbcIgBcFluxVec, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,  
                                      rho_bc_d, u_bc_d, p_bc_d, h_bc_d, 
                                      zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)gamma);

          } 
        

#endif

        }


	void force(double (*rval_dn3)[3], const ForceType f_type) { 

          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
  
          if (f_type==ForceType::TOTAL_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc[ibf],u_bc[ibf],p_bc[ibf],h_bc[ibf],gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc[ibf],u_bc[ibf],p_bc[ibf],h_bc[ibf],gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }
	
      };

#undef NUM_LIMIT
      
      class CbcUptBc : public CbcBc {
      public:
	
        double constant_u_bc[3];
	double T_bc;

        // for a time varying blowing/suction applied to 
        // the domain.  characterized with an amplitude 
        // and frequency (no spatial variation, but could 
        // be additionally parsed from here ... ) 

        double amp;
        double freq; 

        CbcUptBc(BfZone* zone_ptr,Param * param) : CbcBc(zone_ptr) {

	  if (mpi_rank == 0) 
            cout << "CbcUptBc: " << param->str() << endl;
	  
          zone_ptr->lb_cost = 50;
	  
          FOR_I3 constant_u_bc[i] = param->getDouble(1+i);
	  p_bc = param->getDouble(4);
	  T_bc = param->getDouble(5);

          amp  = 0.0;
          freq = 0.0;

          if ( param->size() > 6 ) { 

            int iarg = 6;
            while ( iarg < param->size()) { 

              const string token = param->getUCString(iarg++);
              if ( (token == "AMP") || (token == "AMPLITUDE")) { 
                
                amp = param->getDouble(iarg++);

              } else if ( (token == "FREQ") || (token == "FREQUENCY")) { 

                freq = param->getDouble(iarg++);

              } 

            } 

            // in order to ensure no net mass flow, we insist that 
            // the frequency and amplitude are non-zero ... 

            if ( amp == 0.0) { 

              CERR(" > failed to specify a valid amplitude in CBC_UPT " );

            }

            if ( freq == 0.0) { 

              CERR( " > failed to specify a valid frequency in CBC_UPT"); 

            } 

          }

	}
	
        void setPerturbedVelocityField() { 

          if ( amp > 0.0) { 
         
            assert( freq > 0.0);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 
              for (int i = 0; i < 3; ++i) { 

                // apply the blowing/suction only in the normal 
                // direction ... 

                const double n_mag = MAG(zone_ptr->n_bf[ibf]);
                for (int i = 0; i < 3; ++i) { 
                  double unit_ni = zone_ptr->n_bf[ibf][i]/n_mag;
                  u_bc[ibf][i]   = constant_u_bc[i] + amp*unit_ni*sin(2.0*M_PI*freq*time);
                }

              }
            }
          }
        }

        void initData() {
	  
          rho_bc = p_bc/(R_gas*T_bc);
	  h_bc = R_gas*T_bc*gamma/(gamma-1.0);
	  
          assert(u_bc == NULL);
          u_bc = new double[zone_ptr->nbf][3];

          if ( amp == 0.0) { 
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 u_bc[ibf][i] = constant_u_bc[i];
            }
          } else { 
            setPerturbedVelocityField();
          }
          
#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::initAndCopyToDevice(u_bc_d, (double*)u_bc, 3*zone_ptr->nbf);
#endif

	}

        void rkStep(const double* rk_wgt, const int rk_stage) { 

          if ( (rk_stage == 1 ) && ( amp > 0.0) ) { 
            setPerturbedVelocityField();
          } 
        } 

        void rkStepGpu(const double * rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( (rk_stage == 1) && ( amp > 0.0) && (zone_ptr->nbf > 0)) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;


            assert( u_bc_d);
            assert( zone_ptr->n_bf_d);

            LAUNCH_GPU_KERNEL( setPerturbedVelocityFieldK, grid_size, block_size, 
                               u_bc_d, 
                               (cti_ffp) constant_u_bc[0], 
                               (cti_ffp) constant_u_bc[1], 
                               (cti_ffp) constant_u_bc[2], 
                               (cti_ffp) amp, 
                               (cti_ffp) freq, 
                               (cti_ffp) time, 
                               zone_ptr->n_bf_d, 
                               zone_ptr->nbf);

          } 

#endif
        } 

        void preLoadBalance() {
	  
          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
#endif

	}

        void postLoadBalance() {
	  initData();
	}
     


      };

      class CbcMptBc : public CbcBc {
      public:
	
        double mdot_bc;
	double T_bc;

        CbcMptBc(BfZone* zone_ptr,Param * param) : CbcBc(zone_ptr) {
	  if (mpi_rank == 0) cout << "CbcMptBc: " << param->str() << endl;
	  zone_ptr->lb_cost = 50;
	  mdot_bc = param->getDouble(1);
	  p_bc = param->getDouble(2);
	  T_bc = param->getDouble(3);
	}
	
        void initData() {

	  rho_bc = p_bc/T_bc/R_gas;
          h_bc = gamma*p_bc/((gamma-1.0)*rho_bc); // ideal gas
	  assert(u_bc == NULL);
	  u_bc = new double[zone_ptr->nbf][3];
	  double un_bc  = -mdot_bc/(rho_bc*zone_ptr->area_global);
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    FOR_I3 u_bc[ibf][i] = un_bc*zone_ptr->n_bf[ibf][i]/MAG(zone_ptr->n_bf[ibf]);
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
#endif

	}
	
        void preLoadBalance() {

          // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
	  delete[] u_bc;
	  u_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;
#endif

	}

        void postLoadBalance() {
	  initData();
	} 
        
      };

      class CbcProfileBc : public CbcBc {
      private:
	bool b_filename;
	string filename;
        string fileformat;
	bool b_rho;
	bool b_p;
      public:

        double * p_bc_prof;
        double * h_bc_prof;

        cti_ffp *p_bc_prof_d;
        cti_ffp *h_bc_prof_d;
	
        CbcProfileBc(BfZone* zone_ptr,Param * param) : CbcBc(zone_ptr) {
	  if (mpi_rank == 0) cout << "CbcProfileBc: " << param->str() << endl;

          fileformat = "FLUENT_PROFILE";

	  b_filename = false;
	  b_rho = false;
	  b_p = false;

          p_bc_prof = NULL;
          h_bc_prof = NULL;

          p_bc_prof_d = NULL;
          h_bc_prof_d = NULL;

	  int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ((token == "FILE")||(token == "FILENAME")||(token == "FILE_NAME")) {
	      filename = param->getString(iarg++);
              if (mpi_rank == 0) cout << " > file name: " << filename << endl;
	      // check that the file is available and readable...
	      if (!utils::fileExists(filename)) {
		CERR("PROFILE FILE " << filename << " does not exist");
	      }
	      b_filename = true;
	    }
            else if (token == "FORMAT") {
              fileformat = param->getUCString(iarg++);
              if (mpi_rank == 0) cout << " > file format: " << fileformat << endl;
            }
	    else if (token == "RHO") {
	      rho_bc = param->getDouble(iarg++);
	      COUT(" > RHO " << rho_bc);
	      b_rho = true;
	    }
	    else if (token == "P") {
	      p_bc = param->getDouble(iarg++);
	      COUT(" > P " << p_bc);
	      b_p = true;
	    }
	    else {
	      CERR("unrecognized CBC_PROFILE token: " << token);
	    }
	  }

          if ((fileformat != "FLUENT_PROFILE")&&(fileformat != "ASCII")) {
            CERR("CBC_PROFILE expects FORMAT {FLUENT_PROFILE,ASCII}, where FLUENT_PROFILE is default");
          }
	  // make sure atleast 2 state variables are set...
	  if (!(b_rho&&b_p)) {
	    CERR("CBC_PROFILE state under-specified. Add RHO <rho> P <p>");
	  }
	}

        ~CbcProfileBc() {
          DELETE(u_bc);
          DELETE(p_bc_prof);
          DELETE(h_bc_prof);

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;

          cti::utils::freeOnDevice(p_bc_prof_d); p_bc_prof_d = NULL;

          cti::utils::freeOnDevice(h_bc_prof_d); h_bc_prof_d = NULL;
#endif
        }
	
        void initData() {
	
          assert(b_rho);
	  assert(b_p);
	  h_bc = gamma*p_bc/((gamma-1.0)*rho_bc); // ideal gas
	  assert(u_bc == NULL);
	  u_bc = new double[zone_ptr->nbf][3];

          assert(p_bc_prof == NULL);
          p_bc_prof = new double[zone_ptr->nbf];

          assert(h_bc_prof == NULL);
          h_bc_prof = new double[zone_ptr->nbf];

          // read profiles from input file
          utils::ProfileReader profile;
          profile.init(filename,fileformat);

          profile.ensureVar("u");
          profile.ensureVar("v");
          profile.ensureVar("w");

          //if (profile.getType() == utils::LINE_R) {
          //  profile.setPoints(r_bf,zone_ptr->nbf);
          //}
          if (profile.getType() == utils::LINE_WD) {
            // need to pass wall-distance for each ibf
            double * wall_dist = new double[zone_ptr->nbf];
            CtiData *dn = eval("distance(\"wall\")",true);
            double *dn_data = dn->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              wall_dist[ibf] = dn_data[zone_ptr->cvobf[ibf]]; 
            }
            profile.setPoints(wall_dist,zone_ptr->nbf);
            DELETE(wall_dist);
          }
          else if (profile.getType() == utils::POINT_3D) {
            profile.setPoints(zone_ptr->x_bf,zone_ptr->x_global,zone_ptr->nbf);
          }
          else {
            assert(profile.getType() == utils::LINE_R);
            CERR("LINE_R profile type currently not supported");
          }

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            u_bc[ibf][0] = profile.getData(ibf,"u");
            u_bc[ibf][1] = profile.getData(ibf,"v");
            u_bc[ibf][2] = profile.getData(ibf,"w");

            p_bc_prof[ibf] = p_bc - 0.5*rho_bc*MAG2(u_bc[ibf]);
            h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);
          }

	  // // read the columns of the file...
	  // assert(b_filename);
	  // double *y_profile = NULL;
	  // double *u_profile = NULL;
	  // int n_profile;
	  // int ierr = utils::xcol(y_profile,u_profile,n_profile,filename,1,2);
	  // if (ierr != 0) {
	  //   CERR("cannot parse columns 1 and 2 in profile file: " << filename);
	  // }
	  // for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	  //   const double y_bc = zone_ptr->x_bf[ibf][1];
	  //   if (y_bc <= y_profile[0]) {
	  //     assert(0);
	  //   }
	  //   else if (y_bc >= y_profile[n_profile-1]) {
	  //     u_bc[ibf][0] = u_profile[n_profile-1];
	  //     u_bc[ibf][1] = 0.0;
	  //     u_bc[ibf][2] = 0.0;
	  //   }
	  //   else {
	  //     // bisection...
	  //     int left = 0;
	  //     int right = n_profile-1;
	  //     while ((right - left) > 1) {
	  //       const int middle = (left + right)/2;   // equivalent to floor..
	  //       if (y_bc >= y_profile[middle])
	  //         left = middle;
	  //       else
	  //         right = middle;
	  //     }
	  //     const double w0 = (y_profile[right]-y_bc)/(y_profile[right]-y_profile[left]);
	  //     u_bc[ibf][0] = w0*u_profile[left] + (1.0-w0)*u_profile[right];
	  //     u_bc[ibf][1] = 0.0;
	  //     u_bc[ibf][2] = 0.0;
	  //   }
	  // }
	  // delete[] y_profile;
	  // delete[] u_profile;
	
#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::initAndCopyToDevice(u_bc_d, (double*)u_bc, 3*zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(p_bc_prof_d, p_bc_prof, zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(h_bc_prof_d, h_bc_prof, zone_ptr->nbf);
#endif

        }
        
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            double flux[5];
	    const int icv = zone_ptr->cvobf[ibf];
	    calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
			    rho[icv],u[icv],p[icv],h[icv],gamma,
			    rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc_prof[ibf],gamma);
	    //  mass flux needs to be stored for potential passive scalars..
	    //mf[ibf] = flux[0];
	    FOR_I5 rhs[icv][i] -= flux[i];
	  }
	
        }

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addCbcProfIgBcFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,  
                                      (cti_ffp)rho_bc, u_bc_d, p_bc_prof_d, 
                                      h_bc_prof_d, zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)gamma) ; 


          } 
        

#endif

        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 

          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
  
          if (f_type==ForceType::TOTAL_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }
	
        void preLoadBalance() {
	  // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
          assert(p_bc_prof);
	  delete[] u_bc;
	  u_bc = NULL;
          delete[] p_bc_prof;
          p_bc_prof = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;

          cti::utils::freeOnDevice(p_bc_prof_d); p_bc_prof_d = NULL;
#endif

	}
	
        void postLoadBalance() {
	  initData();
	}

      };

      class CbcZeroVortBc : public CbcBc {
      private:
	bool b_filename;
	string filename;
        string fileformat;
	bool b_rho;
	bool b_p;
      public:

        double dt_filt;

        double * dvdx_bc_prof;
        double * u_cv_filt;

        double * p_bc_prof;
        double * h_bc_prof;

        cti_ffp *dvdx_bc_prof_d;
        cti_ffp *u_cv_filt_d;

        cti_ffp *p_bc_prof_d;
        cti_ffp *h_bc_prof_d;
	
        CbcZeroVortBc(BfZone* zone_ptr,Param * param) : CbcBc(zone_ptr) {
	  if (mpi_rank == 0) cout << "CbcZeroVortBc: " << param->str() << endl;

          fileformat = "FLUENT_PROFILE";

	  b_filename = false;
	  b_rho = false;
	  b_p = false;

          dt_filt = 0.0;

          dvdx_bc_prof = NULL;
          u_cv_filt = NULL; registerData(u_cv_filt, zone_ptr->nbf,zone_ptr->name+":u_cv_filt",READ_DATA|WRITE_DATA);

          registerFunc(&CbcZeroVortBc::u_bc_func,this,zone_ptr->name+":u_bc");
          registerFunc(&CbcZeroVortBc::p_bc_func,this,zone_ptr->name+":p_bc");

          p_bc_prof = NULL;
          h_bc_prof = NULL;

          dvdx_bc_prof_d = NULL;
          u_cv_filt_d = NULL;

          p_bc_prof_d = NULL;
          h_bc_prof_d = NULL;

	  int iarg = 1;
	  while (iarg < param->size()) {
	    string token = param->getUCString(iarg++);
	    if ((token == "FILE")||(token == "FILENAME")||(token == "FILE_NAME")) {
	      filename = param->getString(iarg++);
              if (mpi_rank == 0) cout << " > file name: " << filename << endl;
	      // check that the file is available and readable...
	      if (!utils::fileExists(filename)) {
		CERR("PROFILE FILE " << filename << " does not exist");
	      }
	      b_filename = true;
	    }
            else if (token == "FORMAT") {
              fileformat = param->getUCString(iarg++);
              if (mpi_rank == 0) cout << " > file format: " << fileformat << endl;
            }
	    else if (token == "RHO") {
	      rho_bc = param->getDouble(iarg++);
	      COUT(" > RHO " << rho_bc);
	      b_rho = true;
	    }
	    else if (token == "P") {
	      p_bc = param->getDouble(iarg++);
	      COUT(" > P " << p_bc);
	      b_p = true;
	    }
            else if ((token == "T_FILT")||(token == "DT_FILT")||(token == "T_AVG")||(token == "DT_AVG")) {
              dt_filt = param->getDouble(iarg++);
            }
	    else {
	      CERR("unrecognized CBC_ZERO_VORT token: " << token);
	    }
	  }

          if ((fileformat != "FLUENT_PROFILE")&&(fileformat != "ASCII")) {
            CERR("CBC_ZERO_VORT expects FORMAT {FLUENT_PROFILE,ASCII}, where FLUENT_PROFILE is default");
          }
	  // make sure atleast 2 state variables are set...
	  if (!(b_rho&&b_p)) {
	    CERR("CBC_ZERO_VORT state under-specified. Add RHO <rho> P <p>");
	  }
	}

        ~CbcZeroVortBc() {
          DELETE(u_bc);
          DELETE(dvdx_bc_prof);
          DELETE(p_bc_prof);
          DELETE(h_bc_prof);

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;

          cti::utils::freeOnDevice(dvdx_bc_prof_d); dvdx_bc_prof_d = NULL;

          cti::utils::freeOnDevice(p_bc_prof_d); p_bc_prof_d = NULL;

          cti::utils::freeOnDevice(h_bc_prof_d); h_bc_prof_d = NULL;
#endif
        }

        CtiData *u_bc_func(vector<CtiData*>& argVec,const bool b_eval) {
          int comp = -1;
          if (argVec.size() != 1) {
            eval_cerr << "u_bc() takes 1 argument: got " << argVec.size();
            return NULL;
          }
          if (argVec[0]->getDatatype() != I_DATA) {
            eval_cerr << "u_bc() argument expects I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
            return NULL;
          }
          comp = argVec[0]->i();
          if ( (comp < 0) || (comp >= 3) ) {
            eval_cerr << "u_bc() integer argument should be in [0:3), got " << comp << endl;
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) {
            double *rval_dn = rval->dn();
            assert( (comp >= 0)&&(comp < 3) );
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              rval_dn[ibf] = u_bc[ibf][comp];
            }
          }
          return rval;
        }

        CtiData *p_bc_func(vector<CtiData*>& argVec,const bool b_eval) {
          if (argVec.size() != 0) {
            eval_cerr << "p_bc() takes no arguments: got " << argVec.size();
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              rval_dn[ibf] = p_bc_prof[ibf];
            }
          }
          return rval;
          
        }
	
        void initData() {
	
          assert(b_rho);
	  assert(b_p);
	  h_bc = gamma*p_bc/((gamma-1.0)*rho_bc); // ideal gas
	  assert(u_bc == NULL);
	  u_bc = new double[zone_ptr->nbf][3];

          assert(dvdx_bc_prof == NULL);
          dvdx_bc_prof = new double[zone_ptr->nbf];

          assert(u_cv_filt);
          if ( !checkInit(u_cv_filt,zone_ptr->nbf)) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv = zone_ptr->cvobf[ibf];
              u_cv_filt[ibf] = u[icv][0];
            }
          }

          assert(p_bc_prof == NULL);
          p_bc_prof = new double[zone_ptr->nbf];

          assert(h_bc_prof == NULL);
          h_bc_prof = new double[zone_ptr->nbf];

          // read profiles from input file
          utils::ProfileReader profile;
          profile.init(filename,fileformat);

          profile.ensureVar("u");
          profile.ensureVar("v");
          profile.ensureVar("w");

          profile.ensureVar("dvdx");

          //if (profile.getType() == utils::LINE_R) {
          //  profile.setPoints(r_bf,zone_ptr->nbf);
          //}
          if (profile.getType() == utils::LINE_WD) {
            // need to pass wall-distance for each ibf
            double * wall_dist = new double[zone_ptr->nbf];
            CtiData *dn = eval("distance(\"wall\")",true);
            double *dn_data = dn->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              wall_dist[ibf] = dn_data[zone_ptr->cvobf[ibf]]; 
            }
            profile.setPoints(wall_dist,zone_ptr->nbf);
            DELETE(wall_dist);
          }
          else if (profile.getType() == utils::POINT_3D) {
            profile.setPoints(zone_ptr->x_bf,zone_ptr->x_global,zone_ptr->nbf);
          }
          else {
            assert(profile.getType() == utils::LINE_R);
            CERR("LINE_R profile type currently not supported");
          }

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            u_bc[ibf][0] = profile.getData(ibf,"u");
            u_bc[ibf][1] = profile.getData(ibf,"v");
            u_bc[ibf][2] = profile.getData(ibf,"w");

            dvdx_bc_prof[ibf] = profile.getData(ibf,"dvdx");

            p_bc_prof[ibf] = p_bc - 0.5*rho_bc*MAG2(u_bc[ibf]);
            h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);
          }
	
#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::initAndCopyToDevice(u_bc_d, (double*)u_bc, 3*zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(dvdx_bc_prof_d, dvdx_bc_prof, zone_ptr->nbf);

          assert(u_cv_filt_d == NULL);
          cti::utils::initAndCopyToDevice(u_cv_filt_d, u_cv_filt, zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(p_bc_prof_d, p_bc_prof, zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(h_bc_prof_d, h_bc_prof, zone_ptr->nbf);
#endif

        }
        
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            double flux[5];
	    const int icv = zone_ptr->cvobf[ibf];

            // filter u_cv
            const double eps = dt/(dt+dt_filt);
            u_cv_filt[ibf] = (1.0-eps)*u_cv_filt[ibf] + eps*u[icv][0];

            // set du/dy=dV/dx; dw/dy=0
            const double delta_y = x_bf[ibf][1] - x_cv[icv][1];
            u_bc[ibf][0] = u_cv_filt[ibf] + delta_y*dvdx_bc_prof[ibf];
            if (u_bc[ibf][0] < 0.0) u_bc[ibf][0] = 0.0; //XXX
            //u_bc[ibf][0] = u[icv][0]; // XXX symmetry condition
            //u_bc[ibf][2] = u[icv][2];
            u_bc[ibf][2] = 0.0;
            // recompute pressure and enthalpy
            //p_bc_prof[ibf] = p_bc - 0.5*rho_bc*MAG2(u_bc[ibf]);
            //h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);

            // enforce dp/dn = 0
            //         dh/dn = 0 ?? XXX
            p_bc_prof[ibf] = p[icv];
            h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);

	    calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
			    rho[icv],u[icv],p[icv],h[icv],gamma,
			    rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc_prof[ibf],gamma);
	    //  mass flux needs to be stored for potential passive scalars..
	    //mf[ibf] = flux[0];
	    FOR_I5 rhs[icv][i] -= flux[i];
	  }
	
        }

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            const double eps = dt/(dt+dt_filt);

            LAUNCH_GPU_KERNEL_STREAM( addCbcZeroVortIgBcFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,  
                                      (cti_ffp)rho_bc, u_bc_d, dvdx_bc_prof_d,
                                      u_cv_filt_d,
                                      p_bc_prof_d, h_bc_prof_d, zone_ptr->cvobf_d, 
                                      zone_ptr->x_bf_d, x_cv_d,
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)p_bc, (cti_ffp)gamma, (cti_ffp)eps) ; 


          } 
        

#endif

        }

        void force(double (*rval_dn3)[3], const ForceType f_type) { 

          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
  
          if (f_type==ForceType::TOTAL_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }
	
        void preLoadBalance() {
	  // we could register u_bc and not have to worry about manually
	  // handling this...
	  assert(u_bc);
          assert(dvdx_bc_prof);
          assert(p_bc_prof);
          assert(h_bc_prof);
	  delete[] u_bc;
	  u_bc = NULL;
          delete[] dvdx_bc_prof;
          dvdx_bc_prof = NULL;
          delete[] p_bc_prof;
          p_bc_prof = NULL;
          delete[] h_bc_prof;
          h_bc_prof = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(u_bc_d); u_bc_d = NULL;

          cti::utils::freeOnDevice(dvdx_bc_prof_d); dvdx_bc_prof_d = NULL;

          cti::utils::freeOnDevice(u_cv_filt_d); u_cv_filt_d = NULL;

          cti::utils::freeOnDevice(p_bc_prof_d); p_bc_prof_d = NULL;

          cti::utils::freeOnDevice(h_bc_prof_d); h_bc_prof_d = NULL;
#endif

	}
	
        void postLoadBalance() {
	  initData();
	}

        void syncToHost() {
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::copyToHost(u_cv_filt,u_cv_filt_d,zone_ptr->nbf);

#endif
        
        }
      
      };
      
      class NscbcBc : public IdealGasBc { 
      public: 

        // nscbc boundary conditions will advance a state 
        // the specific types of nscbc boundary conditions 
        // will set the rhs values accordingly ... 

        double * rhs_bc;    
        cti_ffp *rhs_bc_d;

        // for simplicity the boundary conditions will register 
        // a full state vector.  it will help with diagnostic 
        // queries at a later time ... 

        double * rho_bc;
        double * p_bc; 
        double * h_bc;
        double (*u_bc)[3]; 

        double * rho_bc_d;
        double * p_bc_d;
        double * h_bc_d;
        double * u_bc_d;

        NscbcBc(BfZone* zone_ptr) : IdealGasBc(zone_ptr) { 

          rhs_bc   = NULL;

          rho_bc = NULL; registerData(rho_bc,zone_ptr->nbf,zone_ptr->name+":rho_bc",READ_DATA|WRITE_DATA);
          p_bc   = NULL; registerData(p_bc  ,zone_ptr->nbf,zone_ptr->name+":p_bc"  ,READ_DATA|WRITE_DATA);
          h_bc   = NULL; registerData(h_bc  ,zone_ptr->nbf,zone_ptr->name+":h_bc"  ,READ_DATA|WRITE_DATA);
          u_bc   = NULL; registerData(u_bc  ,zone_ptr->nbf,zone_ptr->name+":u_bc"  ,READ_DATA|WRITE_DATA);

          // device memory ptrs... may not be accessed.

          rhs_bc_d = NULL;
          rho_bc_d = NULL;
          p_bc_d   = NULL;
          h_bc_d   = NULL;
          u_bc_d   = NULL;

        } 

        virtual ~NscbcBc() { 

          DELETE(rhs_bc);
          DELETE(p_bc);
          DELETE(rho_bc);
          DELETE(h_bc);
          DELETE(u_bc);

          //DELETE_DEVICE(rhs_bc_d);
          //DELETE_DEVICE(p_bc_d);
          //DELETE_DEVICE(rho_bc_d);
          //DELETE_DEVICE(h_bc_d);
          //DELETE_DEVICE(u_bc_d);

        } 


        void addBoundaryFlux(double(*rhs)[5], const int rk_stage) { 

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 
            
            double flux[5];
            const int icv = zone_ptr->cvobf[ibf];

            calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
			    rho[icv],u[icv],p[icv],h[icv],gamma,
			    rho_bc[ibf],u_bc[ibf],p_bc[ibf],h_bc[ibf],gamma);

            for (int i = 0; i < 5; ++i) 
              rhs[icv][i] -= flux[i];

          }

        } 

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addCbcIgBcFluxVec, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,  
                                      rho_bc_d, u_bc_d, p_bc_d, h_bc_d, 
                                      zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)gamma) ; 


          } 
        
#endif

        }

        void initData() { 

          assert( rho_bc);
          assert( p_bc);
          assert( h_bc);
          assert( u_bc);

          assert ( rhs_bc == NULL);
          rhs_bc = new double[zone_ptr->nbf*5*NRK];

          if ( !checkInit(rho_bc,zone_ptr->nbf)) { 

            if ( mpi_rank == 0 ) 
              cout << " > resetting rho_bc for zone: " << zone_ptr->getName() << endl;

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const int icv = zone_ptr->cvobf[ibf];
              rho_bc[ibf]   = rho[icv];

            }

          }

          if ( !checkInit(u_bc,zone_ptr->nbf)) { 

            if ( mpi_rank == 0 ) 
              cout << " > resetting u_bc for zone: " << zone_ptr->getName() << endl;


            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const int icv = zone_ptr->cvobf[ibf];
              for (int i = 0; i < 3; ++i) 
                u_bc[ibf][i]   = u[icv][i];

            }

          }


          if ( !checkInit(p_bc,zone_ptr->nbf)) { 

            // XXX assuming that the primitive data has been set 
            // at this point ... for p, h... 

            if ( mpi_rank == 0 ) 
              cout << " > resetting p_bc for zone: " << zone_ptr->getName() << endl;


            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const int icv = zone_ptr->cvobf[ibf];
              p_bc[ibf]   = p[icv];

            }

          }


          if ( !checkInit(h_bc,zone_ptr->nbf)) { 

            if ( mpi_rank == 0 ) 
              cout << " > resetting h_bc for zone: " << zone_ptr->getName() << endl;


            // XXX assuming that the primitive data has been set 
            // at this point ... for p, h... 

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const int icv = zone_ptr->cvobf[ibf];
              h_bc[ibf]   = h[icv];

            }

          }


          // initial conditions on the host have all been set... 
          // now we will copy the information to the device if 
          // it has been requested ... 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          cti::utils::initAndCopyToDevice( rho_bc_d, rho_bc       , zone_ptr->nbf);
          cti::utils::initAndCopyToDevice( u_bc_d  ,(double*) u_bc, zone_ptr->nbf*3);
          cti::utils::initAndCopyToDevice( p_bc_d  , p_bc         , zone_ptr->nbf);
          cti::utils::initAndCopyToDevice( h_bc_d  , h_bc         , zone_ptr->nbf);

          cti::utils::initOnDevice( rhs_bc_d, NRK*zone_ptr->nbf*5);

#endif

        }

        void preLoadBalance() { 

          // the boundary condition states are registered, so we dont need 
          // to do anything with those, but we need to take down the device 
          // memory here ... 

          delete[] rhs_bc; rhs_bc = NULL;

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice( rhs_bc_d); rhs_bc_d = NULL;
          cti::utils::freeOnDevice( rho_bc_d); rho_bc_d = NULL;
          cti::utils::freeOnDevice( u_bc_d  ); u_bc_d   = NULL;
          cti::utils::freeOnDevice( p_bc_d  ); p_bc_d   = NULL;
          cti::utils::freeOnDevice( h_bc_d  ); h_bc_d   = NULL;
#endif


        } 


        void postLoadBalance() { 

          // registered data should have reappeared by now... reinit
          initData();

        } 

        void syncToHost() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::copyToHost(p_bc,p_bc_d,zone_ptr->nbf);
          cti::utils::copyToHost(h_bc,h_bc_d,zone_ptr->nbf);
          cti::utils::copyToHost(rho_bc,rho_bc_d,zone_ptr->nbf);
          cti::utils::copyToHost((double*)u_bc,u_bc_d,zone_ptr->nbf*3);

#endif

        } 

        void query() { 

          // grab the static pressure, temperature and the inlet mass flux ...

          double my_buf[4];

          for (int i = 0; i < 4; ++i) 
            my_buf[i] = 0.0;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const int icv = zone_ptr->cvobf[ibf];
            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * p[icv];
            my_buf[2]    += zone_ptr->area_bf[ibf] * T[icv];
            my_buf[3]    += rho[icv] * DOT_PRODUCT( u[icv], zone_ptr->n_bf[ibf]) ;

          } 

          double buf[4];
          MPI_Reduce(my_buf, buf, 4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) { 

            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:p " << buf[1]/buf[0] 
                 << " 10:T " << buf[2]/buf[0] 
                 << " 12:mdot " << buf[3] 
                 << endl; 
          }

        } 

        void force(double (*rval_dn3)[3], const ForceType f_type) {

          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
  
          if (f_type==ForceType::TOTAL_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc[ibf],u_bc[ibf],p_bc[ibf],h_bc[ibf],gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc[ibf],u_bc[ibf],p_bc[ibf],h_bc[ibf],gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }

      };
        

      class NscbcMtBc : public NscbcBc { 
      public:

        double mdot_bc_;
        double T_bc_;
        double swirl_;
	double (*u_fr)[3];

        cti_ffp * u_fr_d;
	
	string help() {
	  return
	    "NSCBC_MT is used for subsonic inlets where the mass and temperature is known. The pressure\n" 
	    "gets set from some other boundary condition or the initial condition. Example:\n"
	    "  <zonename> NSCBC_MT MDOT 0.52 T 273.15 [SWIRL <double in degrees>]";
	}
	
        NscbcMtBc(BfZone* zone_ptr, Param* param) : NscbcBc(zone_ptr) { 

          int ierr = 0;
          int iarg = 1;

          swirl_   = 0.0;

          while ( iarg < param->size()) { 

            string token = utils::toUpperCase( param->getString(iarg++));
	    if ( token == "MDOT") { 
	      mdot_bc_ = param->getDouble(iarg++);
              ierr    |= (1<<0); 
            } 
            else if ( token == "T") { 
              T_bc_    = param->getDouble(iarg++);
              ierr    |= (1<<1);
            } 
            else if ( token == "SWIRL") { 
              // this is an optional parameter -- defaulted to zero.
              swirl_   = param->getDouble(iarg++);
            } 
            else { 
              CERR("unrecognized NSCBC_MT token: " << token << "\n" << help());
	    } 
          }

          // using bits in ierr to keep track of what has been set ... 

          if ( ierr != 3 ) { 
	    CERR( "NSCBC_MT parsing problem. Please fix syntax\n" << help());
          }
	  
	  u_fr   = NULL;
          u_fr_d = NULL;

          // swirl is specified in degrees ... 

          if ( (swirl_ < -90.0) && (swirl_ > 90.0)) { 
            CERR( " > invalid specification of SWIRL in NSCBC_MT \n \
                      must be between -90 < swirl < 90 degrees ");
          } 

          swirl_ *= M_PI / 180.0;

        }

	~NscbcMtBc() {
	  DELETE(u_fr);
	}

	void initData() {

          NscbcBc::initData();

	  assert( u_fr == NULL);
	  u_fr = new double[zone_ptr->nbf][3];
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
	    for (int i = 0; i < 3; ++i)
	      u_fr[ibf][i] = 0.0;
	  
	  if ( frame_rotation != NULL) {
            double r[3];
	    for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];
	      u_fr[ibf][0] = -frame_rotation[1]*r[2] +
		              frame_rotation[2]*r[1];

	      u_fr[ibf][1] = -frame_rotation[2]*r[0] +
		              frame_rotation[0]*r[2];

	      u_fr[ibf][2] = -frame_rotation[0]*r[1] +
		              frame_rotation[1]*r[0];
	    }
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::initAndCopyToDevice(u_fr_d,(double*)u_fr,3*zone_ptr->nbf);
#endif

	}

        void rkStep(const double *rk_wgt, const int rk_stage) {
          
          // the cv state is frozen right now -- so we can 
          // calc the rhs for the rk update at the same time.. 

          // note that the sign convention for this bc is inward pointing... 

          const double rhoun_bc = -mdot_bc_ / zone_ptr->area_global;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            // step 1: populate the rhs ... in this boundary 
            // condition, there is only one unspecified characteristic
            // which gets populated in rhs[0] -- recall rhs_bc 
            // has been flattened to a single ptr... 

            const int icv = zone_ptr->cvobf[ibf];

            double unit_n[3];
            double mag_n = MAG(zone_ptr->n_bf[ibf]);
            for (int i = 0; i < 3; ++i) 
              unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

            const double delta_bf = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf]; 
            
            calcNscbcIgSubsonicInletRhs( &rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf], 
                                         unit_n, delta_bf, 
                                         rho_bc[ibf], u_bc[ibf], p_bc[ibf], 
                                         rho[icv]   , u[icv]   , p[icv], gamma); 

            // step 2: advance the rho... 

            double rhs_rho_agg = 0.0; 

            for (int irk = 0; irk < rk_stage; ++irk) { 
              rhs_rho_agg += rk_wgt[irk] * rhs_bc[irk*zone_ptr->nbf*5+5*ibf];
            }

            rho_bc[ibf]       += dt * rhs_rho_agg;

            
            // now update everyone else .. 

	    double ut[3];
	    double ufr_n = DOT_PRODUCT(u_fr[ibf],unit_n);
	    for (int i = 0; i < 3; ++i)
	      ut[i] = u_fr[ibf][i] - ufr_n*unit_n[i];

	    for (int i =0; i < 3; ++i)
	      u_bc[ibf][i] = (rhoun_bc*unit_n[i] + rho_bc[ibf]*ut[i])/rho_bc[ibf];

            // lastly add in the swirl component to the velocity field.  
            // we add a component to the velocity field that is locally 
            // orthogonal to the face normal.. 

            const double us_bc    = rhoun_bc / rho_bc[ibf] * tan(swirl_);
            double v_r[3]         = DIFF(zone_ptr->x_bf[ibf], zone_ptr->x_global);
            const double dp       = DOT_PRODUCT( v_r, unit_n);
            for (int i = 0; i < 3; ++i) 
              v_r[i] -= dp*unit_n[i]; // remove the normal project.. 

            const double us_vec[3] = CROSS_PRODUCT(v_r,unit_n);
            const double us_mag    = MAG(us_vec);

            if ( us_mag > 0.0) { 
              
              for (int i = 0; i < 3 ; ++i) 
                u_bc[ibf][i]        += us_bc*us_vec[i]/us_mag;

            } 

            p_bc[ibf] = R_gas * T_bc_  * rho_bc[ibf];
            h_bc[ibf] = gamma/(gamma-1.0) * R_gas * T_bc_;

	    // the coriolis force here is not relevant because there is only
	    // 1 characteristic equation being solved (which is for rho)

          }
        
        }

        void rkStepGpu(const double *rk_wgt, const int rk_stage) {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          // note that the sign convention for this bc is inward pointing... 

          const cti_ffp rhoun_bc = -mdot_bc_ / zone_ptr->area_global;

          if (zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL( rkStepIgNscbcSubsonicInletBc, grid_size, 
                               block_size, rhs_bc_d, rho_bc_d, 
                               u_bc_d, p_bc_d, h_bc_d, 
                               sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                               u_fr_d,
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d,
                               zone_ptr->area_bf_d,zone_ptr->area_over_delta_bf_d,
                               zone_ptr->x_bf_d,  zone_ptr->x_global_d,
                               rk_wgt,rk_stage,
                               zone_ptr->nbf, (cti_ffp)gamma, (cti_ffp)R_gas,
                               rhoun_bc,(cti_ffp)T_bc_,(cti_ffp)swirl_,dt);  

          } 

#endif

        }
        
        void preLoadBalance() {

          NscbcBc::preLoadBalance();
	  
          delete[] u_fr; u_fr = NULL; 
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice( u_fr_d); u_fr_d = NULL;

#endif
        
        }
	
        void postLoadBalance() {
	  initData();
	}
      };

      class NscbcOutletMdotBc : public NscbcBc { 
      public: 

        double rhou_target;
        double * rhou_avg_d;
        double T_relax;

        NscbcOutletMdotBc(BfZone* zone_ptr, Param * param) : NscbcBc(zone_ptr) { 

          if ( mpi_rank == 0 ) 
            cout << " NscbcOutletMdotBc: " << param->str() << endl;
          
          zone_ptr->lb_cost = 50;

          rhou_avg_d  = NULL;

          int iarg    = 1;
          int ierr    = 0;
          T_relax     = -1.0;
          rhou_target = -1.0; 

          while ( iarg < param->size()) { 

            string token = utils::toUpperCase( param->getString(iarg++));

            if ( token == "T_RELAX") { 
              
              T_relax = param->getDouble(iarg++);
              ierr |= (1<<0);

            } else if ( token == "MDOT" ) { 

              rhou_target = param->getDouble(iarg++);
              ierr  |= (1<<1);

            } else { 

              CERR(" > unrecognized token in NSCBC_OUTLET_MDOT: " << token);

            } 

          } 

          if ( ierr != 3 ) { 
            CERR (" > NSCBC_OUTLET_MDOT requires params: MDOT <double> T_RELAX <double>");
          }

          rhou_target /= zone_ptr->area_global;
          
          if ( mpi_rank == 0 ) 
            cout << " zone: " << zone_ptr->getName() << " rhou target   " << rhou_target 
                 << " T relax : " << T_relax << endl;


        } 

        void initData() { 

          NscbcBc::initData();

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          
          assert( rhou_avg_d == NULL);
          cti::utils::initOnDevice(rhou_avg_d,3);

#endif

        }

        void rkStep(const double *rk_wgt, const int rk_stage) { 

          // compute the present mdot ..

          double my_buf[3] = {0.0,0.0,0.0};

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const double bf_area = MAG(zone_ptr->n_bf[ibf]);
            const double un_dA   = DOT_PRODUCT(u_bc[ibf],zone_ptr->n_bf[ibf]);
            my_buf[0]           += bf_area;
            my_buf[1]           += rho_bc[ibf]*un_dA;
            my_buf[2]           += un_dA;

          }

          double buf[3];
          MPI_Allreduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,mpi_comm);
          double rhou_avg = buf[1]/buf[0];
          double u_avg    = buf[2]/buf[0];

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const double bf_area = MAG(zone_ptr->n_bf[ibf]);
            double unit_n[3]     = {zone_ptr->n_bf[ibf][0]/bf_area,
                                    zone_ptr->n_bf[ibf][1]/bf_area,
                                    zone_ptr->n_bf[ibf][2]/bf_area};

            const int icv      = zone_ptr->cvobf[ibf];
            const double delta = bf_area/zone_ptr->area_over_delta_bf[ibf];
    
            calcNscbcIgOutletMdotWeakRhs(&rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf],
                                         unit_n,delta,
                                         rho_bc[ibf],u_bc[ibf],p_bc[ibf],
                                         rho[icv],u[icv],p[icv],
                                         gamma,rhou_avg,u_avg,rhou_target, T_relax);

            if ( frame_rotation != NULL) {

	      // term 1: -omega x omega x r
	      // centrifugal acceleration
              double r[3];
              FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];

	      double coeff[3], rot_accel[3];
	      coeff[0] = frame_rotation[1]*r[2] - frame_rotation[2]*r[1];
	      coeff[1] = frame_rotation[2]*r[0] - frame_rotation[0]*r[2];
	      coeff[2] = frame_rotation[0]*r[1] - frame_rotation[1]*r[0];

	      rot_accel[0] = -(frame_rotation[1]*coeff[2] - frame_rotation[2]*coeff[1]);
	      rot_accel[1] = -(frame_rotation[2]*coeff[0] - frame_rotation[0]*coeff[2]);
	      rot_accel[2] = -(frame_rotation[0]*coeff[1] - frame_rotation[1]*coeff[0]);

	      // term 2: -2 omega x v
	      // Coriolis acceleration

	      rot_accel[0] -= 2.0*(frame_rotation[1]*u_bc[ibf][2] - frame_rotation[2]*u_bc[ibf][1]);
	      rot_accel[1] -= 2.0*(frame_rotation[2]*u_bc[ibf][0] - frame_rotation[0]*u_bc[ibf][2]);
	      rot_accel[2] -= 2.0*(frame_rotation[0]*u_bc[ibf][1] - frame_rotation[1]*u_bc[ibf][0]);

	      for (int i = 0; i <3; ++i)
		rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf+i+2] += rot_accel[i];
		
	    }

            // we are updating all 5 here ... 

            double rhs_agg[5];
            for (int i = 0; i < 5; ++i) {  
              rhs_agg[i] = 0.0; 
            } 

            for (int irk = 0; irk < rk_stage; ++irk) { 
              for (int i = 0; i < 5; ++i) 
                rhs_agg[i] += rk_wgt[irk] * rhs_bc[irk*zone_ptr->nbf*5+5*ibf+i];
            }

            rho_bc[ibf]       += dt * rhs_agg[0];
            p_bc[ibf]         += dt * rhs_agg[1];
            for (int i = 0; i < 3; ++i) 
              u_bc[ibf][i]    += dt * rhs_agg[2+i];


            // everyone is advanced, we just need to set h_bc for posterity .. 

            h_bc[ibf]         = gamma/(gamma-1.0) * p_bc[ibf] / rho_bc[ibf];

          }
 
        } 

        void rkStepGpu(const double *rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          // everyone needs to go into the beginning bc we need 
          // to reduce the mass flux ... 

          cti::utils::deviceMemset(rhou_avg_d,0,3*sizeof(double));

          { 

            const int block_size = 128;
            const int grid_size  = 128; 

            LAUNCH_GPU_KERNEL( (computeMdotStuff<cti_ffp,block_size>), 
                               grid_size, block_size, rhou_avg_d, 
                               rho_bc_d,u_bc_d,u_f_d,sp_vol_f_d,
                               zone_ptr->cvobf_d,zone_ptr->n_bf_d,
                               zone_ptr->nbf);

          }

          double my_buf[3], buf[3];
          cti::utils::copyToHost(my_buf,rhou_avg_d,3);
            
          MPI_Allreduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,mpi_comm);

          double rhou_avg = buf[1]/buf[0];
          double u_avg    = buf[2]/buf[0];

          if ( (mpi_rank == 0) && (step%sync_interval == 0) && (rk_stage ==1))
            cout << " zone : " << zone_ptr->getName() << " rhou : " << rhou_avg 
                 << "   rhou_target : " << rhou_target << endl;

          if ( zone_ptr->nbf > 0 ) {

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1)/ block_size;

            LAUNCH_GPU_KERNEL( rkStepIgNscbcOutletMdotBc, 
                               grid_size, block_size, 
                               rhs_bc_d, rho_bc_d, u_bc_d, p_bc_d, h_bc_d,
                               sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                               zone_ptr->x_bf_d, frame_rotation_d,
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d,
                               zone_ptr->area_bf_d,zone_ptr->area_over_delta_bf_d,
                               rk_wgt,rk_stage,
                               cti_ffp(rhou_avg),
                               cti_ffp(u_avg),
                               cti_ffp(rhou_target),
                               cti_ffp(T_relax),
                               cti_ffp(gamma), dt,
                               zone_ptr->nbf);

          } 

#endif

        } 
 
        void preLoadBalance() {

          NscbcBc::preLoadBalance();

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice(rhou_avg_d);
          rhou_avg_d = NULL;

#endif

        }
	
        void postLoadBalance() {
	  initData();
	}

      };

      class NscbcOutletMdotUpdateBc : public NscbcBc { 
      public: 

        double rhou_target;
        double * rhou_avg_d;
        double T_relax;
        double p_ref;
        double p_gain;
        double rhou_running;
        double rhou_wgt;
	double L_ref_;
	double sigma_;

	string help() {
	  return
	    "NSCBC_OUTLET_MDOT_UPDATE applies the Navier-Stokes-Characteristic-Boundary-Condition where\n"
	    "the mass flow rate is known at an outlet. It adjusts the outlet reference pressure to match\n"
	    "the mass flow rate to the target. It requires a mass flow rate MDOT, a time scale T_RELAX to\n"
	    "control how frequently the reference pressure is adjusted, a characteristic length scale \n"
	    "L_REF, an optional parameter GAIN with default 1.0 to control the magnitude of pressure \n"
	    "adjustment, and an optional relaxation SIGMA with default 0.2. Examples:\n"
	    "  <zonename> NSCBC_OUTLET_MDOT_UPDATE MDOT 20.0 T_RELAX 0.2 L_REF 0.1\n"
	    "  <zonename> NSCBC_OUTLET_MDOT_UPDATE MDOT 20.0 T_RELAX 0.2 GAIN 2.0 L_REF 0.1\n"
	    "  <zonename> NSCBC_OUTLET_MDOT_UPDATE MDOT 20.0 T_RELAX 0.2 GAIN 2.0 L_REF 0.1 SIGMA 0.1";
	}
	
        NscbcOutletMdotUpdateBc(BfZone* zone_ptr, Param * param) : NscbcBc(zone_ptr) { 

          if ( mpi_rank == 0 ) 
            cout << " NscbcOutletMdotUpdateBc: " << param->str() << endl;
          
          zone_ptr->lb_cost = 50;

          rhou_avg_d  = NULL;

          int iarg    = 1;
          int ierr    = 0;
          T_relax     = -1.0;
          rhou_target = -1.0;
          p_gain      = 1.0; 

	  sigma_      = 0.2;
	  
          p_ref        = -1.0;
          rhou_running = -1.0;
          rhou_wgt     =  0.0;
          registerData(p_ref,zone_ptr->name+":p_ref", READ_DATA|WRITE_DATA);
          registerData(rhou_running,zone_ptr->name+":rhou_running",READ_DATA|WRITE_DATA);
          registerData(rhou_wgt    ,zone_ptr->name+":rhou_wgt",READ_DATA|WRITE_DATA);

          while ( iarg < param->size()) { 

            string token = utils::toUpperCase( param->getString(iarg++));

            if ( token == "T_RELAX") { 
              
              T_relax = param->getDouble(iarg++);
              ierr |= (1<<0);

            } else if ( token == "MDOT" ) { 

              rhou_target = param->getDouble(iarg++);
              ierr  |= (1<<1);

            } else if ( token == "GAIN") { 

              p_gain      = param->getDouble(iarg++);

            } else if ( (token == "L_REF") || (token == "L")) { 

              L_ref_ = param->getDouble(iarg++);
              ierr  |= (1<<2);

            } else if ( token == "SIGMA" ) { 

              sigma_  = param->getDouble(iarg++);

            }
	    else { 

              CERR(" > unrecognized token in NSCBC_OUTLET_MDOT: " << token);

            } 

          } 

          if ( ierr != 7 ) { 
            CERR (" > NSCBC_OUTLET_MDOT requires params: MDOT <double> T_RELAX <double> L_REF <double>");
          }

          rhou_target /= zone_ptr->area_global;
          
          if ( mpi_rank == 0 ) 
            cout << " zone: " << zone_ptr->getName() << " rhou target   " << rhou_target 
                 << " T relax : " << T_relax << " L_ref : " << L_ref_ << " sigma : " << sigma_ << endl;


        } 

        void initData() { 

          NscbcBc::initData();

          if ( p_ref < 0.0) { 

            // these should all have been registered together ... 

            assert( rhou_running < 0.0); 
            assert( rhou_wgt == 0.0);

            // hasnt been set yet... build it based on the average
            // of the state on the interior.. 

            double my_buf[2] = { 0.0, 0.0}; 
            double buf[2]; 

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

              const int icv = zone_ptr->cvobf[ibf];
              my_buf[0]    += zone_ptr->area_bf[ibf];
              my_buf[1]    += zone_ptr->area_bf[ibf]*p[icv];

            } 

            MPI_Allreduce(my_buf,buf,2,MPI_DOUBLE,MPI_SUM,mpi_comm);

            p_ref        = buf[1]/buf[0];
            rhou_running = 0.0;
            rhou_wgt     = 0.0;

          } else { 

            if ( mpi_rank == 0 ) { 

              cout << zone_ptr->getName() << " p_ref = " << p_ref << endl;
              cout << zone_ptr->getName() << " rhou = " << rhou_running << endl;
              cout << zone_ptr->getName() << " rhou_wgt = " << rhou_wgt << endl;

            } 

          }



#if defined(BUILD_CUDA) || defined(BUILD_HIP) 
          
          assert( rhou_avg_d == NULL);
          cti::utils::initOnDevice(rhou_avg_d,3);

#endif

        }

        void rkStep(const double *rk_wgt, const int rk_stage) { 

          if ( rk_stage == 1 ) { 

            // compute the present mdot ..

            double my_buf[3]     = {0.0,0.0,0.0};
            const double sos_ref = sqrt(gamma*p_ref/rho_ref);

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

              const double bf_area = MAG(zone_ptr->n_bf[ibf]);
              const double un_dA   = DOT_PRODUCT(u_bc[ibf],zone_ptr->n_bf[ibf]);
              my_buf[0]           += bf_area;
              my_buf[1]           += rho_bc[ibf]*un_dA;
              my_buf[2]           += un_dA;

            }

            double buf[3];
            MPI_Allreduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,mpi_comm);
            double rhou_avg = buf[1]/buf[0];
            //double u_avg    = buf[2]/buf[0];

            // update the running average total 

            rhou_running = (rhou_wgt*rhou_running + dt*rhou_avg)/(rhou_wgt + dt);
            rhou_wgt    += dt;

            if ( rhou_wgt >= T_relax) { 

              // we have entered a new epoch for the pressure... 
              if ( mpi_rank == 0) { 
                cout << " step  time  rhou_target  rhou_act  p_ref = " << step << "    " << time << "    "
		     << rhou_target << "    " << rhou_running << "     " << p_ref << endl;
              } 


              // update the p_ref here ... 

              p_ref += p_gain*(rhou_running - rhou_target)*sos_ref;

              if ( mpi_rank == 0 ) 
                cout << zone_ptr->getName() << " ; new p_ref = " << p_ref << endl;


              // reset

              rhou_running = 0.0;
              rhou_wgt     = 0.0;


            }

          } 
          
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            // step 1: populate the rhs ... 

            const int icv = zone_ptr->cvobf[ibf];

            double unit_n[3];
            double mag_n = MAG(zone_ptr->n_bf[ibf]);
            for (int i = 0; i < 3; ++i) 
              unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

            const double delta_bf = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf]; 

	    /*
            calcNscbcIgOutletRhs2(&rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf],
                                 unit_n, delta_bf,
                                 rho_bc[ibf], u_bc[ibf], p_bc[ibf], 
                                 rho[icv], u[icv], p[icv], gamma, 
                                 T_relax, p_ref);
	    */
	    
	    calcNscbcIgOutletRhs(&rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf],
                                 unit_n, delta_bf,
                                 rho_bc[ibf], u_bc[ibf], p_bc[ibf], 
                                 rho[icv], u[icv], p[icv], gamma, L_ref_, sigma_, p_ref);

	    //if ( frame_rotation != NULL) 
	    if ( false) {

	      // term 1: -omega x omega x r
	      // centrifugal acceleration
              double r[3];
              FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];

	      double coeff[3], rot_accel[3];
	      coeff[0] = frame_rotation[1]*r[2] - frame_rotation[2]*r[1];
	      coeff[1] = frame_rotation[2]*r[0] - frame_rotation[0]*r[2];
	      coeff[2] = frame_rotation[0]*r[1] - frame_rotation[1]*r[0];

	      rot_accel[0] = -(frame_rotation[1]*coeff[2] - frame_rotation[2]*coeff[1]);
	      rot_accel[1] = -(frame_rotation[2]*coeff[0] - frame_rotation[0]*coeff[2]);
	      rot_accel[2] = -(frame_rotation[0]*coeff[1] - frame_rotation[1]*coeff[0]);

	      // term 2: -2 omega x v
	      // Coriolis acceleration

	      rot_accel[0] -= 2.0*(frame_rotation[1]*u_bc[ibf][2] - frame_rotation[2]*u_bc[ibf][1]);
	      rot_accel[1] -= 2.0*(frame_rotation[2]*u_bc[ibf][0] - frame_rotation[0]*u_bc[ibf][2]);
	      rot_accel[2] -= 2.0*(frame_rotation[0]*u_bc[ibf][1] - frame_rotation[1]*u_bc[ibf][0]);

	      for (int i = 0; i <3; ++i)
		rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf+i+2] += rot_accel[i];
		
	    }

            // we are updating all 5 here ... 

            double rhs_agg[5];
            for (int i = 0; i < 5; ++i) {  
              rhs_agg[i] = 0.0; 
            } 

            for (int irk = 0; irk < rk_stage; ++irk) { 
              for (int i = 0; i < 5; ++i) 
                rhs_agg[i] += rk_wgt[irk] * rhs_bc[irk*zone_ptr->nbf*5+5*ibf+i];
            }

            rho_bc[ibf]       += dt * rhs_agg[0];
            p_bc[ibf]         += dt * rhs_agg[1];
            for (int i = 0; i < 3; ++i) 
              u_bc[ibf][i]    += dt * rhs_agg[2+i];


            // everyone is advanced, we just need to set h_bc for posterity .. 

            h_bc[ibf]         = gamma/(gamma-1.0) * p_bc[ibf] / rho_bc[ibf];

          }
 
        } 

        void rkStepGpu(const double *rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( rk_stage == 1 ) { 

            // everyone needs to go into the beginning bc we need 
            // to reduce the mass flux ... 

            cti::utils::deviceMemset(rhou_avg_d,0,3*sizeof(double));

            { 

              const int block_size = 128;
              const int grid_size  = 128; 

              LAUNCH_GPU_KERNEL( (computeMdotStuff<cti_ffp,block_size>), 
                                 grid_size, block_size, rhou_avg_d, 
                                 rho_bc_d,u_bc_d,u_f_d,sp_vol_f_d,
                                 zone_ptr->cvobf_d,zone_ptr->n_bf_d,
                                 zone_ptr->nbf);

            }

            double my_buf[3], buf[3];
            cti::utils::copyToHost(my_buf,rhou_avg_d,3);
            
            MPI_Allreduce(my_buf,buf,3,MPI_DOUBLE,MPI_SUM,mpi_comm);

            double rhou_avg = buf[1]/buf[0];
            //double u_avg    = buf[2]/buf[0];

            // update the running average total 

            rhou_running = (rhou_wgt*rhou_running + dt*rhou_avg)/(rhou_wgt + dt);
            rhou_wgt    += dt;

            if ( rhou_wgt >= T_relax) { 

              // we have entered a new epoch for the pressure... 
              if ( mpi_rank == 0) {
		cout << " step  time  rhou_target  rhou_act  p_ref = " << step << "    " << time << "    "
		     << rhou_target << "    " << rhou_running << "     " << p_ref << endl;
              } 


              // update the p_ref here ... 

              const double sos_ref = sqrt(gamma*p_ref/rho_ref);
              p_ref += p_gain*(rhou_running - rhou_target)*sos_ref;

              if ( mpi_rank == 0 ) 
                cout << zone_ptr->getName() << " ; new p_ref = " << p_ref << endl;


              // reset

              rhou_running = 0.0;
              rhou_wgt     = 0.0;


            }

          }

          if (zone_ptr->nbf > 0) { 

            assert( rhs_bc_d);

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

	    /*
            LAUNCH_GPU_KERNEL( rkStepIgNscbcSubsonicOutletBc2, grid_size, 
                               block_size, rhs_bc_d, rho_bc_d, 
                               u_bc_d, p_bc_d, h_bc_d, 
                               sp_vol_f_d, u_f_d, p_f_d, h_f_d, 
                               zone_ptr->x_bf_d, frame_rotation_d,
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d,
                               zone_ptr->area_bf_d,
                               zone_ptr->area_over_delta_bf_d,
                               rk_wgt,rk_stage,
                               zone_ptr->nbf, (cti_ffp)gamma, (cti_ffp)R_gas,
                               (cti_ffp) p_ref, (cti_ffp) T_relax,dt); 
	    */

	    LAUNCH_GPU_KERNEL( rkStepIgNscbcSubsonicOutletBc, grid_size, 
                               block_size, rhs_bc_d, rho_bc_d, 
                               u_bc_d, p_bc_d, h_bc_d, 
                               sp_vol_f_d, u_f_d, p_f_d, h_f_d, 
                               zone_ptr->x_bf_d, frame_rotation_d,
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d,
                               zone_ptr->area_bf_d,zone_ptr->area_over_delta_bf_d,
                               rk_wgt,rk_stage,
                               zone_ptr->nbf, (cti_ffp)gamma, (cti_ffp)R_gas,
                               (cti_ffp) p_ref, (cti_ffp) L_ref_, (cti_ffp) sigma_, dt);  

          }



#endif

        } 
 
        void preLoadBalance() {

          NscbcBc::preLoadBalance();

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          cti::utils::freeOnDevice(rhou_avg_d);
          rhou_avg_d = NULL;

#endif

        }
	
        void postLoadBalance() {
	  initData();
	}

      };


      class NscbcOutletPressureBc : public NscbcBc { 
      public: 

        double p_bc_;
        double L_ref_;
        double sigma_;

	string help() {
	  return
	    "NSCBC_OUTLET_P applies the Navier-Stokes-Characteristic-Boundary-Condition where\n"
	    "the pressure is known at an outlet. It requires a pressure P_REF, a characteristic\n"
	    "length scale L_REF, and an optional relaxation SIGMA with default 0.2. Examples:\n"
	    "  <zonename> NSCBC_OUTLET_P P_REF 101325 L_REF 1.0\n"
	    "  <zonename> NSCBC_OUTLET_P P_REF 101325 L_REF 1.0 SIGMA 0.1";
	}
	
	
        NscbcOutletPressureBc(BfZone* zone_ptr, Param* param) : NscbcBc(zone_ptr) { 
	  
	  if (mpi_rank == 0) cout << "NscbcOutletPressureBc: " << param->str() << endl;
          zone_ptr->lb_cost = 50;

          int ierr = 0;
          int iarg = 1;
          sigma_   = 0.2;

          while ( iarg < param->size()) { 

            string token = utils::toUpperCase( param->getString(iarg++));
 
            if ( token == "P_REF") { 

              p_bc_ = param->getDouble(iarg++);
              ierr |= (1<<0);

            } 
            else if ( (token == "L_REF") || (token == "L")) { 

              L_ref_ = param->getDouble(iarg++);
              ierr  |= (1<<1);

            } 
            else if ( token == "SIGMA" ) { 

              sigma_  = param->getDouble(iarg++);

            } 
            else { 

              CERR("unrecognized NSCBC_OUTLET_P token: " << token << "\n" << help());

            } 

          }
	  
          if ( ierr != 3 ) { 

	    CERR( "NSCBC_OUTLET_P parsing problem. Please fix syntax\n" << help());
	    
          } 


        } 

        void initData() { 

          NscbcBc::initData();

        }

        void rkStep(const double *rk_wgt, const int rk_stage) { 

          // the cv state is frozen right now -- so we can 
          // calc the rhs for the rk update at the same time.. 

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            // step 1: populate the rhs ... 

            const int icv = zone_ptr->cvobf[ibf];

            double unit_n[3];
            double mag_n = MAG(zone_ptr->n_bf[ibf]);
            for (int i = 0; i < 3; ++i) 
              unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

            const double delta_bf = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf]; 

            calcNscbcIgOutletRhs(&rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf],
                                 unit_n, delta_bf,
                                 rho_bc[ibf], u_bc[ibf], p_bc[ibf], 
                                 rho[icv], u[icv], p[icv], gamma, L_ref_, sigma_, p_bc_);

	    //if ( frame_rotation != NULL) 
            if ( false ) { 

	      // term 1: -omega x omega x r
	      // centrifugal acceleration
              double r[3];
              FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];

	      double coeff[3], rot_accel[3];
	      coeff[0] = frame_rotation[1]*r[2] - frame_rotation[2]*r[1];
	      coeff[1] = frame_rotation[2]*r[0] - frame_rotation[0]*r[2];
	      coeff[2] = frame_rotation[0]*r[1] - frame_rotation[1]*r[0];

	      rot_accel[0] = -(frame_rotation[1]*coeff[2] - frame_rotation[2]*coeff[1]);
	      rot_accel[1] = -(frame_rotation[2]*coeff[0] - frame_rotation[0]*coeff[2]);
	      rot_accel[2] = -(frame_rotation[0]*coeff[1] - frame_rotation[1]*coeff[0]);

	      // term 2: -2 omega x v
	      // Coriolis acceleration

	      rot_accel[0] -= 2.0*(frame_rotation[1]*u_bc[ibf][2] - frame_rotation[2]*u_bc[ibf][1]);
	      rot_accel[1] -= 2.0*(frame_rotation[2]*u_bc[ibf][0] - frame_rotation[0]*u_bc[ibf][2]);
	      rot_accel[2] -= 2.0*(frame_rotation[0]*u_bc[ibf][1] - frame_rotation[1]*u_bc[ibf][0]);

	      for (int i = 0; i <3; ++i)
		rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf+i+2] += rot_accel[i];
		
	    }

            // we are updating all 5 here ... 

            double rhs_agg[5];
            for (int i = 0; i < 5; ++i) {  
              rhs_agg[i] = 0.0; 
            } 

            for (int irk = 0; irk < rk_stage; ++irk) { 
              for (int i = 0; i < 5; ++i) 
                rhs_agg[i] += rk_wgt[irk] * rhs_bc[irk*zone_ptr->nbf*5+5*ibf+i];
            }

            rho_bc[ibf]       += dt * rhs_agg[0];
            p_bc[ibf]         += dt * rhs_agg[1];
            for (int i = 0; i < 3; ++i) 
              u_bc[ibf][i]    += dt * rhs_agg[2+i];


            // everyone is advanced, we just need to set h_bc for posterity .. 

            h_bc[ibf]         = gamma/(gamma-1.0) * p_bc[ibf] / rho_bc[ibf];

          }
 
        } 

        void rkStepGpu(const double *rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if (zone_ptr->nbf > 0) { 

            assert( rhs_bc_d);

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL( rkStepIgNscbcSubsonicOutletBc, grid_size, 
                               block_size, rhs_bc_d, rho_bc_d, 
                               u_bc_d, p_bc_d, h_bc_d, 
                               sp_vol_f_d, u_f_d, p_f_d, h_f_d, 
                               zone_ptr->x_bf_d, frame_rotation_d,
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d,
                               zone_ptr->area_bf_d,zone_ptr->area_over_delta_bf_d,
                               rk_wgt,rk_stage,
                               zone_ptr->nbf, (cti_ffp)gamma, (cti_ffp)R_gas,
                               (cti_ffp) p_bc_, (cti_ffp) L_ref_, (cti_ffp) sigma_, dt);  

          } 


#endif

        } 
        
        void preLoadBalance() {

          NscbcBc::preLoadBalance();
        
        }
	
        void postLoadBalance() {
	  initData();
	}

      };


      class NscbcOutletPressureProfileBc : public NscbcBc { 
      public: 

        double* p_ref;
	double* r_bf;

	double L_ref_;
        double sigma_;

	cti_ffp* p_ref_d;
	
	string filename;
        string fileformat;
        double xc[3],axis[3];
	
	string help() {
	  return
	    "NSCBC_OUTLET_PRESSURE_PROFILE applies the Navier-Stokes-Characteristic-Boundary-Condition where\n"
	    "the pressure is known at an outlet. It requires a file containing pressure distribution, a\n"
	    "characteristic length scale L_REF, and an optional relaxation SIGMA with default 0.2. Examples:\n"
	    "  <zonename> NSCBC_OUTLET_PRESSURE_PROFILE FILE <filename> FORMAT FLUENT_PROFILE L_REF 1.0\n"
	    "  <zonename> NSCBC_OUTLET_PRESSURE_PROFILE FILE <filename> FORMAT ASCII L_REF 1.0 SIGMA 0.1\n"
	    "  <zonename> NSCBC_OUTLET_PRESSURE_PROFILE FILE <filename> FORMAT ASCII XC <x0> <y0> <z0> AXIS <n0> <n1> <n2> L_REF 1.0\n";
	}
	
	
        NscbcOutletPressureProfileBc(BfZone* zone_ptr, Param* param) : NscbcBc(zone_ptr) { 
	  
	  if (mpi_rank == 0) cout << "NscbcOutletPressureProfileBc: " << param->str() << endl;
          zone_ptr->lb_cost = 50;

          sigma_   = 0.2;

	  p_ref   = NULL;
	  p_ref_d = NULL;

	  r_bf    = NULL;
	  
	  fileformat = "FLUENT_PROFILE";
          FOR_I3 xc[i] = zone_ptr->x_global[i];
          FOR_I3 axis[i] = -zone_ptr->n_global[i]; // point into domain
	  const double mag = sqrt(DOT_PRODUCT(axis,axis));
	  assert ( mag > 0.0) ;
	  FOR_I3 axis[i] /= mag ;   // make sure it is a unit vector

          int ierr = 0;
          int iarg = 1;

          while ( iarg < param->size()) { 

            string token = utils::toUpperCase( param->getString(iarg++));
 
            if ( token == "FILE") {

	      filename = param->getString(iarg++);
	      if (mpi_rank == 0) cout << " > file name: " << filename << endl;
              ierr |= (1<<0);

            }
	    else if (token == "FORMAT") {

	      fileformat = param->getUCString(iarg++);
	      if (mpi_rank == 0) cout << " > file format: " << fileformat << endl;
	      
	    }
	    else if (token == "XC") {
	      
              FOR_I3 xc[i] = param->getDouble(iarg++);

	    }
            else if (token == "AXIS") {
	      
              FOR_I3 axis[i] = param->getDouble(iarg++);
	      const double mag = sqrt(DOT_PRODUCT(axis,axis));
              assert ( mag > 0.0) ;
              FOR_I3 axis[i] /= mag ;   // make sure it is a unit vector
	      
            }
	    
            else if ( (token == "L_REF") || (token == "L")) { 

              L_ref_ = param->getDouble(iarg++);
              ierr  |= (1<<1);

            } 
            else if ( token == "SIGMA" ) { 

              sigma_  = param->getDouble(iarg++);

            } 
            else { 

              CERR("unrecognized NSCBC_OUTLET_PRESSURE_PROFILE token: " << token << "\n" << help());

            } 

          }
	  
          if ( ierr != 3 ) { 

	    CERR( "NSCBC_OUTLET_PRESSURE_PROFILE parsing problem. Please fix syntax\n" << help());
	    
          } 


        }

	~NscbcOutletPressureProfileBc() {

	  DELETE(p_ref);
	  DELETE(r_bf);

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
	  if (p_ref_d) cti::utils::freeOnDevice(p_ref_d);
#endif
	  
	}

        void initData() {

          NscbcBc::initData();

	  assert(p_ref == NULL); p_ref = new double[zone_ptr->nbf];

	  // read profiles from input file
          utils::ProfileReader profile;
          profile.init(filename,fileformat);
          profile.ensureVar("pressure");

	  // calc r_bf
	  assert(r_bf == NULL);  r_bf = new double[zone_ptr->nbf];
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    double dx[3] = DIFF(zone_ptr->x_bf[ibf],xc);
	    const double dxn = DOT_PRODUCT(dx,axis);
	    FOR_I3 dx[i] -= dxn*axis[i];
	    r_bf[ibf] = MAG(dx);
	  }

	  if (profile.getType() == utils::LINE_R) {
            profile.setPoints(r_bf,zone_ptr->nbf);
          }
          else if (profile.getType() == utils::LINE_WD) {
            // need to pass wall-distance for each ibf
            double * wall_dist = new double[zone_ptr->nbf];
            CtiData *dn = eval("distance(\"wall\")",true);
            double *dn_data = dn->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              wall_dist[ibf] = dn_data[zone_ptr->cvobf[ibf]];
            }
            profile.setPoints(wall_dist,zone_ptr->nbf);
            DELETE(wall_dist);
          }
          else {
            assert(profile.getType() == utils::POINT_3D);
            profile.setPoints(zone_ptr->x_bf,zone_ptr->x_global,zone_ptr->nbf);
          }

	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
	    p_ref[ibf] = profile.getData(ibf,"pressure");
	  }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          assert(p_ref_d == NULL);
          cti::utils::initAndCopyToDevice(p_ref_d,p_ref,zone_ptr->nbf);
#endif
        }

        void rkStep(const double *rk_wgt, const int rk_stage) { 

          // the cv state is frozen right now -- so we can 
          // calc the rhs for the rk update at the same time.. 

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 

            // step 1: populate the rhs ... 

            const int icv = zone_ptr->cvobf[ibf];

            double unit_n[3];
            double mag_n = MAG(zone_ptr->n_bf[ibf]);
            for (int i = 0; i < 3; ++i) 
              unit_n[i] = zone_ptr->n_bf[ibf][i] / mag_n;

            const double delta_bf = zone_ptr->area_bf[ibf]/zone_ptr->area_over_delta_bf[ibf]; 

            calcNscbcIgOutletRhs(&rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf],
				 unit_n, delta_bf,
				 rho_bc[ibf], u_bc[ibf], p_bc[ibf], 
				 rho[icv], u[icv], p[icv], gamma, L_ref_, sigma_, p_ref[ibf]);

	    //if ( frame_rotation != NULL) 
            if ( false ) { 

	      // term 1: -omega x omega x r
	      // centrifugal acceleration
              double r[3];
              FOR_I3 r[i] = zone_ptr->x_bf[ibf][i] - frame_rotation[i+3];

	      double coeff[3], rot_accel[3];
	      coeff[0] = frame_rotation[1]*r[2] - frame_rotation[2]*r[1];
	      coeff[1] = frame_rotation[2]*r[0] - frame_rotation[0]*r[2];
	      coeff[2] = frame_rotation[0]*r[1] - frame_rotation[1]*r[0];

	      rot_accel[0] = -(frame_rotation[1]*coeff[2] - frame_rotation[2]*coeff[1]);
	      rot_accel[1] = -(frame_rotation[2]*coeff[0] - frame_rotation[0]*coeff[2]);
	      rot_accel[2] = -(frame_rotation[0]*coeff[1] - frame_rotation[1]*coeff[0]);

	      // term 2: -2 omega x v
	      // Coriolis acceleration

	      rot_accel[0] -= 2.0*(frame_rotation[1]*u_bc[ibf][2] - frame_rotation[2]*u_bc[ibf][1]);
	      rot_accel[1] -= 2.0*(frame_rotation[2]*u_bc[ibf][0] - frame_rotation[0]*u_bc[ibf][2]);
	      rot_accel[2] -= 2.0*(frame_rotation[0]*u_bc[ibf][1] - frame_rotation[1]*u_bc[ibf][0]);

	      for (int i = 0; i <3; ++i)
		rhs_bc[(rk_stage-1)*zone_ptr->nbf*5+5*ibf+i+2] += rot_accel[i];
		
	    }

            // we are updating all 5 here ... 

            double rhs_agg[5];
            for (int i = 0; i < 5; ++i) {  
              rhs_agg[i] = 0.0; 
            } 

            for (int irk = 0; irk < rk_stage; ++irk) { 
              for (int i = 0; i < 5; ++i) 
                rhs_agg[i] += rk_wgt[irk] * rhs_bc[irk*zone_ptr->nbf*5+5*ibf+i];
            }

            rho_bc[ibf]       += dt * rhs_agg[0];
            p_bc[ibf]         += dt * rhs_agg[1];
            for (int i = 0; i < 3; ++i) 
              u_bc[ibf][i]    += dt * rhs_agg[2+i];


            // everyone is advanced, we just need to set h_bc for posterity .. 

            h_bc[ibf]         = gamma/(gamma-1.0) * p_bc[ibf] / rho_bc[ibf];

          }
 
        } 

        void rkStepGpu(const double *rk_wgt, const int rk_stage) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if (zone_ptr->nbf > 0) { 

            assert( rhs_bc_d);

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL( rkStepIgNscbcSubsonicOutletProfileBc, grid_size, 
                               block_size, rhs_bc_d, rho_bc_d, 
                               u_bc_d, p_bc_d, h_bc_d, 
                               sp_vol_f_d, u_f_d, p_f_d, h_f_d, 
                               zone_ptr->x_bf_d, frame_rotation_d,
                               zone_ptr->cvobf_d, zone_ptr->n_bf_d,
                               zone_ptr->area_bf_d,zone_ptr->area_over_delta_bf_d,
                               rk_wgt,rk_stage,
                               zone_ptr->nbf, (cti_ffp)gamma, (cti_ffp)R_gas,
                               p_ref_d, (cti_ffp) L_ref_, (cti_ffp) sigma_, dt);  

          } 


#endif

        } 
        
        void preLoadBalance() {

          NscbcBc::preLoadBalance();

	  DELETE(p_ref);
	  DELETE(r_bf);	  
	  
#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::freeOnDevice(p_ref_d); p_ref_d = NULL;
#endif
        }
	
        void postLoadBalance() {
	  initData();
	}

      };

      class RecycleBc : public IdealGasBc {
      public:
	
        // The cbc bcs have to set these vars...
        // only u needs to vary spatially so it can relect the local normal in some cases
	
        double rho_bc;
	double p_bc;
	double h_bc;
	
        double mdot_bc;
        double *u_send;
        double *u_recv;

        cti_ffp * u_send_d;
        cti_ffp * u_recv_d;

        double * dx;
        DataExchanger * exchanger;

        double  (*u_bc)[3];
        cti_ffp* u_bc_d; 

        double * work_arr_host;
        double * work_arr_d;

        RecycleBc(BfZone* zone_ptr, Param * param) : IdealGasBc(zone_ptr) {
	  
          rho_bc   = BIG_DOUBLE;
	  p_bc     = BIG_DOUBLE;
	  h_bc     = BIG_DOUBLE;
          mdot_bc  = BIG_DOUBLE;

          dx       = NULL;
          u_send   = NULL;
          u_recv   = NULL;
          u_send_d = NULL;
          u_recv_d = NULL;

          u_bc     = NULL;
          u_bc_d   = NULL;

          work_arr_host   = NULL;
          work_arr_d      = NULL;

          if ( mpi_rank == 0 ) 
            cout << " RecycleBc: " << param->str() << endl;

          zone_ptr->lb_cost = 200;
          exchanger         = NULL;
          dx                = NULL;

          int iarg = 1;
          while ( iarg < param->size() ) { 

            string token = param->getUCString(iarg++);

            if ( token == "DX") { 

              dx = new double[3];
              for (int i = 0; i < 3; ++i) 
                dx[i] = param->getDouble(iarg++);

            } else if ( token == "MDOT") { 

              mdot_bc = param->getDouble(iarg++);

            } else if ( token == "P") { 

              p_bc    = param->getDouble(iarg++);

            } else if ( token == "RHO") { 

              rho_bc  = param->getDouble(iarg++);

            } else { 

              CERR( " > unrecognized token " << token << 
                     " during parse of of bc:\n" << param->str());

            } 

          }

          if ( dx == NULL) { 
            CERR( " > RECYCLE_BC requires DX <dx0> <dx1> <dx2>"); 
          } 

          if ( mdot_bc == BIG_DOUBLE) { 
            CERR( " > RECYCLE_BC require MDOT <mdot>");
          } 

          if ( rho_bc == BIG_DOUBLE) { 
            CERR( " > RECYCLE_BC requires RHO <rho-bc>");
          } 

          if ( p_bc == BIG_DOUBLE) { 
            CERR( " > RECYCLE_BC requires P <p-bc>");
          } 

        }
	
        ~RecycleBc() {
	  
          DELETE(dx);
          DELETE(u_send);
          DELETE(u_recv);

          DELETE(work_arr_host);
          DELETE(u_bc);

          if ( exchanger) { 
            delete exchanger;
          } 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( u_send_d) { 

            cti::utils::freeOnDevice(u_send_d);
            u_send_d = NULL;

          } 

          if ( u_recv_d) { 

            cti::utils::freeOnDevice(u_recv_d);
            u_recv_d = NULL;

          } 

          if ( work_arr_d) { 
            
            cti::utils::freeOnDevice(work_arr_d);
            work_arr_d = NULL;
          
          } 

          if ( u_bc_d) { 
            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;
          } 


#endif

	}

        void initData() { 


          if ( mpi_rank == 0) { 
            cout << " > init exchanger for zone : " << zone_ptr->getName() << endl;
          }

          assert ( exchanger == NULL); 

          assert( dx);
          double (*x_ex)[3] = new double[zone_ptr->nbf][3];

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) { 
            for (int i = 0; i < 3; ++i) 
              x_ex[ibf][i] = zone_ptr->x_bf[ibf][i] + dx[i];

          }

          exchanger = new DataExchanger();
          exchanger->init(x_ex,zone_ptr->nbf);
         
          // make sure all exchange points are inside domain...

          int * bf_flag = new int[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            bf_flag[ibf] = -1;

          for (int ii =0; ii < exchanger->nx_unpack; ++ii) 
            bf_flag[exchanger->idopt_recv[ii]]++;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            if (bf_flag[ibf] == -1) {
              cout << " > could not find exchange point: " << COUT_VEC(x_ex[ibf]) << ". Please modify DX." << endl;
              cout.flush();
              assert(0);
            }
          }

          delete[] bf_flag;

          // we expect that the communicator is sparse... 
          
          exchanger->buildSparseCommunicatorStruct();
          const int n_recv = exchanger->recv_stuff.size()/3;
          //const int n_send = exchanger->send_stuff.size()/3;

          { 

            // actually lets check the worst case complexity of the 
            // communicator ... 

            int self_count = 0;
            for (int ii = 0; ii < n_recv; ++ii) { 

              int rank = exchanger->recv_stuff[3*ii];
              if ( rank == mpi_rank ) { 
                // this is the self send .. 
                self_count = exchanger->recv_stuff[3*ii+1];
                break;
              }
            }

            double my_fax = double(zone_ptr->nbf-self_count)/double(zone_ptr->nbf);
            double fax;
            MPI_Reduce(&my_fax,&fax,1,MPI_DOUBLE,MPI_MAX,0,mpi_comm);

            if ( mpi_rank == 0 ) 
              cout << " > worst case exch fraction : " << fax << endl;

          } 

          // lastly allocate some data for the vel magnitude .. 
        
          assert( u_send == NULL); u_send = new double[3*exchanger->nx_pack];
          assert( u_recv == NULL); u_recv = new double[3*exchanger->nx_unpack];

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( u_send_d == NULL);
          assert( u_recv_d == NULL);

          cti::utils::initOnDevice(u_send_d,3*exchanger->nx_pack);
          cti::utils::initOnDevice(u_recv_d,3*exchanger->nx_unpack);

#endif 

          delete[] x_ex;

          h_bc = p_bc/rho_bc*gamma/(gamma-1.0);

          assert( u_bc == NULL);
          u_bc = new double[zone_ptr->nbf][3];

          doVelocityExchange();

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          assert( u_bc_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);

          assert( work_arr_d == NULL);
          cti::utils::initOnDevice(work_arr_d,1024);

          assert( work_arr_host == NULL);
          work_arr_host = new double[1024];
#endif

        }

        void preLoadBalance() {

          DELETE(u_bc);
          DELETE(u_send);
          DELETE(u_recv);
          DELETE(work_arr_host);

          if ( exchanger) { 
            delete exchanger;
            exchanger = NULL;
          } 

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( u_bc_d ) { 

            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;

          } 

          if ( u_send_d ) { 

            cti::utils::freeOnDevice(u_send_d);
            u_send_d = NULL;

          } 

          if ( u_recv_d ) { 

            cti::utils::freeOnDevice(u_recv_d);
            u_recv_d = NULL;

          } 

          if (work_arr_d) { 

            cti::utils::freeOnDevice(work_arr_d);
            work_arr_d = NULL;
          } 

#endif

        } 

        void postLoadBalance() { 
          initData();
        }

        void doVelocityExchange() {

          assert ( exchanger);
          const int n_recv = exchanger->recv_stuff.size()/3;
          const int n_send = exchanger->send_stuff.size()/3;

          MPI_Request * recv_req = new MPI_Request[n_recv];
          MPI_Request * send_req = new MPI_Request[n_send];

          int jj_recv      = 0;
          int jj_send      = 0;

          const int UPDATE_TAG = 11112;

          // post the receives first ... 

          int my_recv_disp  = -1;
          int my_self_count = -1;

          for (int ii = 0; ii < n_recv; ++ii) { 

            const int rank = exchanger->recv_stuff[3*ii];
            if ( rank != mpi_rank ) {

              MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                        3*exchanger->recv_stuff[3*ii+1], 
                        MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                        &recv_req[jj_recv]);

              ++jj_recv;

            } else { 

              // this is data i am sending to myself, so record 
              // the displacement 

              my_self_count = exchanger->recv_stuff[3*ii+1];
              my_recv_disp  = exchanger->recv_stuff[3*ii+2];

            }

          }

          // pack the data for the exchange ... 

          for (int ii = 0; ii < exchanger->nx_pack; ++ii) {
            const int icv = exchanger->cvopt[ii];
            assert((icv >= 0)&&(icv < ncv));
            FOR_I3 u_send[3*ii+i] = u[icv][i];
          }

          // now post the sends

          for (int ii = 0; ii < n_send; ++ii) { 

            const int rank = exchanger->send_stuff[3*ii];

            if ( rank != mpi_rank) {

              MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2], 
                         3*exchanger->send_stuff[3*ii+1], 
                         MPI_DOUBLE,rank, 
                         UPDATE_TAG,mpi_comm,&send_req[jj_send]);

              ++jj_send;

            } else { 

              // check the counts for the self copy ... 

              assert( exchanger->send_stuff[3*ii+1] == my_self_count);

            }

          }

          // the communication is in flight right now.  while we wait, 
          // if there are any self communications that present then 
          // we can take of them here 

          if ( my_self_count > 0 ) {

            assert( n_send > 0);
            const int send_disp = exchanger->send_stuff[3*(n_send-1)+2];

            for (int ii = 0; ii < my_self_count; ++ii) {
              FOR_I3 u_recv[3*(my_recv_disp+ii)+i] = u_send[3*(send_disp+ii)+i];
            }

          }

          // now we have to wait ... 

          MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

          // we copy everything that isnt the self comm.  note 
          // that the self comm on the recv side is also at the end... 

          // all the data is now there.  

          if ( exchanger->nx_unpack > 0 ) {

            for (int ii = 0; ii < exchanger->nx_unpack; ++ii) {
              const int ibf = exchanger->idopt_recv[ii];
              FOR_I3 u_bc[ibf][i] = u_recv[3*ii+i];
            }

          }

          // ensure the sends complete ... and we're out.

          MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

          delete[] send_req;
          delete[] recv_req; 

          // calc mf global...

          double my_buf = 0.0;
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            const int icv = zone_ptr->cvobf[ibf];
            my_buf += rho[icv]*DOT_PRODUCT(u_bc[ibf],zone_ptr->n_bf[ibf]);
          }
          double buf;
          MPI_Allreduce(&my_buf,&buf,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

          // rescale...

          assert(buf != 0.0);
          buf = -mdot_bc/buf;
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            FOR_I3 u_bc[ibf][i] *= buf;
          }

        }

        void doVelocityExchangeGpu() {

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert ( exchanger);
          const int n_recv = exchanger->recv_stuff.size()/3;
          const int n_send = exchanger->send_stuff.size()/3;

          MPI_Request * recv_req = new MPI_Request[n_recv];
          MPI_Request * send_req = new MPI_Request[n_send];

          int jj_recv      = 0;
          int jj_send      = 0;

          const int UPDATE_TAG = 11112;

          // post the receives first ... 

          int my_recv_disp  = -1;
          int my_self_count = -1;

          for (int ii = 0; ii < n_recv; ++ii) { 

            const int rank = exchanger->recv_stuff[3*ii];
            if ( rank != mpi_rank ) {

              MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                        3*exchanger->recv_stuff[3*ii+1], 
                        MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                        &recv_req[jj_recv]);

              ++jj_recv;

            } else { 

              // this is data i am sending to myself, so record 
              // the displacement 

              my_self_count = exchanger->recv_stuff[3*ii+1];
              my_recv_disp  = exchanger->recv_stuff[3*ii+2];

            }

          }

          // pack the data for the exchange ... 

          if ( exchanger->nx_pack > 0) { 

            const int block_size = 128;
            const int grid_size  = (exchanger->nx_pack + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL(packVelocityForExchange, 
                              grid_size, block_size, 
                              u_send_d, u_f_d, exchanger->cvopt_d, exchanger->nx_pack);

          }

          // XXX should consider using pinned memory for this transfer ...
          // i do not need to copy the whole send buffer bc some of the 
          // data is going to be local to this rank.... 

          int n_pack = 0;
          if ( n_send > 0 ) { 

            if ( exchanger->send_stuff[3*(n_send-1)] == mpi_rank ) { 

              // the send_disp is the count of the pack that isnt 
              // related to the self send ... 

              n_pack = exchanger->send_stuff[3*(n_send-1)+2]; 

            } else { 

              // i dont have a self send, so i need to whole buffer

              n_pack = exchanger->nx_pack;

            }
          }

          if ( n_pack > 0 ) { 
            cti::utils::copyToHost(u_send,u_send_d,3*n_pack);
          }

          // now post the sends

          for (int ii = 0; ii < n_send; ++ii) { 

            const int rank = exchanger->send_stuff[3*ii];

            if ( rank != mpi_rank) {

              MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2], 
                         3*exchanger->send_stuff[3*ii+1], 
                         MPI_DOUBLE,rank, 
                         UPDATE_TAG,mpi_comm,&send_req[jj_send]);

              ++jj_send;

            } else { 

              // check the counts for the self copy ... 

              assert( exchanger->send_stuff[3*ii+1] == my_self_count);

            }

          }

          // the communication is in flight right now.  while we wait, 
          // if there are any self communications that present then 
          // we can take of them here 

          if ( my_self_count > 0 ) {

            assert( n_send > 0);
            const int send_disp = exchanger->send_stuff[3*(n_send-1)+2];
            const int block_size = 128;
            const int grid_size  = (my_self_count + block_size - 1)/block_size;

            LAUNCH_GPU_KERNEL(performExchangeSelfComm, 
                              grid_size, block_size, 
                              u_recv_d, u_send_d, send_disp,
                              my_recv_disp, my_self_count);

          }

          // now we have to wait ... 

          MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

          // we copy everything that isnt the self comm.  note 
          // that the self comm on the recv side is also at the end... 

          int n_unpack = 0;
          if ( n_recv > 0 ) { 

            if ( exchanger->recv_stuff[3*(n_recv-1)] == mpi_rank ) { 

              // the send_disp is the count of the pack that isnt 
              // related to the self send ... 

              n_unpack = exchanger->recv_stuff[3*(n_recv-1)+2]; 

            } else { 

              // i dont have a self send, so i need to whole buffer

              n_unpack = exchanger->nx_unpack;

            }
          }

          if ( n_unpack > 0 ) { 
            cti::utils::copyToDevice(u_recv_d,u_recv, 3*n_unpack);
          }

          // all the data is now there.  

          if ( exchanger->nx_unpack > 0 ) {

            const int block_size   = 128;
            const int grid_size    = (exchanger->nx_unpack + block_size - 1)/block_size;

            LAUNCH_GPU_KERNEL( unpackVelocityForExchange, 
                               grid_size, block_size,
                               u_bc_d, u_recv_d, exchanger->idopt_recv_d, exchanger->nx_unpack);

          }

          // ensure the sends complete ... and we're out.

          MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

          delete[] send_req;
          delete[] recv_req; 

          // calc mf global...


          if (zone_ptr->nbf > 0) {

            { 

              const int block_size = 128;
              const int grid_size  = 128; // must be smaller than max block size...  

              LAUNCH_GPU_KERNEL( mf_bc_reduce_block2, grid_size, block_size, 
                                 work_arr_d, rho_d, u_bc_d, zone_ptr->n_bf_d, 
                                 zone_ptr->cvobf_d, zone_ptr->nbf);

            } 

            { 

              // reduce in place... 

              const int block_size = 128; // must equal grid_size above.. 
              const int grid_size  =   1;

              cti::utils::reduce(block_size,block_size,grid_size,
                                 work_arr_d,work_arr_d); 

              // we need to copy back the value in red_work_arr[0] to the host
              // and then pass through the MPI reduce
              cti::utils::copyToHost(work_arr_host,work_arr_d,1); 

            } 
            
          }
          else {
            work_arr_host[0] = 0.0;
          }

          double tmp;
          MPI_Allreduce(work_arr_host,&tmp,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

          if (zone_ptr->nbf > 0) {

            assert(tmp != 0.0);
            tmp = -mdot_bc/tmp;

            { 

              const int block_size = 256;
              const int grid_size  = (3*zone_ptr->nbf + block_size - 1) / block_size; 

              LAUNCH_GPU_KERNEL( sax, grid_size, block_size,
                                 u_bc_d, tmp, 0, 3*zone_ptr->nbf);

            } 

          }

#endif

        }



        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
	  
	  for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            double flux[5];
	    const int icv = zone_ptr->cvobf[ibf];
	    calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
			    rho[icv],u[icv],p[icv],h[icv],gamma,
			    rho_bc,u_bc[ibf],p_bc,h_bc,gamma);
	    //  mass flux needs to be stored for potential passive scalars..
	    //mf[ibf] = flux[0];
	    FOR_I5 rhs[icv][i] -= flux[i];
	  }
	
        }

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage, 
                                 gpuStream_t& stream) { 

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) { 

            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addCbcIgBcFlux, grid_size, 
                                      block_size, stream, rhs, 
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,  
                                      (cti_ffp)rho_bc, u_bc_d, (cti_ffp) p_bc, 
                                      (cti_ffp)h_bc, zone_ptr->cvobf_d, 
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)gamma) ; 


          } 
        

#endif

        }

        void rkStep(const double *rk_wgt,const int rk_stage) {
        
          if ( rk_stage == 1) { 
            doVelocityExchange();
          }
        
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {
        
          if ( rk_stage == 1) { 
            doVelocityExchangeGpu();
          }
        
        }
        void syncToHost() {}

        void query() { 

          // grab the static pressure, temperature and the inlet mass flux ...

          double my_buf[4];

          for (int i = 0; i < 4; ++i) 
            my_buf[i] = 0.0;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {

            const int icv = zone_ptr->cvobf[ibf];
            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * p[icv];
            my_buf[2]    += zone_ptr->area_bf[ibf] * T[icv];
            my_buf[3]    += rho[icv] * DOT_PRODUCT( u[icv], zone_ptr->n_bf[ibf]) ;

          } 

          double buf[4];
          MPI_Reduce(my_buf, buf, 4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) { 

            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step 
                 << " 6:time " << time 
                 << " 8:p " << buf[1]/buf[0] 
                 << " 10:T " << buf[2]/buf[0] 
                 << " 12:mdot " << buf[3] 
                 << endl;

          }

        } 


        void force(double (*rval_dn3)[3], const ForceType f_type) { 

          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) 
            FOR_I3 rval_dn3[ibf][i] = 0.0;
  
          if (f_type==ForceType::TOTAL_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc,h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc,h_bc,gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i]; 
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE){
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf){
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }
      
      };

      class RecycleRescaleBc : public IdealGasBc {
      public:

        double rho_bc;
        double p_bc;
        double * p_bc_prof;
        double h_bc;
        double * h_bc_prof;
        double u_edge;

        double delta_in;
        double delta_re;
        double re_delta_in;
        
        double y_bc_zero;
        double y_orientation;

        double t_mean;

        double mdot_bc;
        double *u_send;
        double *u_recv;

        cti_ffp * u_send_d;
        cti_ffp * u_recv_d;

        double lz;
        double ly;
        double *dx;
        double (*x_ex)[3];
        double (*mean_u_ex)[3];
        cti_ffp * mean_u_ex_d;
        DataExchanger * exchanger;

        double (*u_bc)[3];
        cti_ffp * u_bc_d;
        cti_ffp * p_bc_prof_d;
        cti_ffp * h_bc_prof_d;

        bool b_filename;
        string filename;
        string fileformat;
        double (*mean_u_bc)[3];
        cti_ffp * mean_u_bc_d;

        double * u_scale;
        cti_ffp * u_scale_d;

        double * work_arr_host;
        double * work_arr_d;

        RecycleRescaleBc(BfZone* zone_ptr, Param * param) : IdealGasBc(zone_ptr) {
        
          rho_bc      = BIG_DOUBLE;
          p_bc        = BIG_DOUBLE;
          h_bc        = BIG_DOUBLE;
          mdot_bc     = BIG_DOUBLE;
          lz          = BIG_DOUBLE;
          ly          = BIG_DOUBLE;
          delta_in    = BIG_DOUBLE;
          delta_re    = BIG_DOUBLE;
          re_delta_in = BIG_DOUBLE;
          u_edge      = BIG_DOUBLE;

          y_bc_zero = 0.0;
          y_orientation = 1.0;

          dx          = NULL;
          x_ex        = NULL;
          u_send      = NULL;
          u_recv      = NULL;
          u_send_d    = NULL;
          u_recv_d    = NULL;

          u_bc        = NULL;
          u_bc_d      = NULL;
          p_bc_prof   = NULL;
          p_bc_prof_d = NULL;
          h_bc_prof   = NULL;
          h_bc_prof_d = NULL;

          mean_u_ex   = NULL;
          mean_u_ex_d = NULL;

          fileformat = "ASCII";

          b_filename  = false;
          mean_u_bc   = NULL;
          mean_u_bc_d = NULL;

          u_scale     = NULL;
          u_scale_d   = NULL;

          work_arr_host = NULL;
          work_arr_d    = NULL;

          registerData(u_bc,zone_ptr->nbf,zone_ptr->name+":u_bc", DEP_DATA); // do not read/write
          registerData(mean_u_ex,zone_ptr->nbf,zone_ptr->name+":mean_u_ex", READ_DATA|WRITE_DATA);
          registerFunc(&RecycleRescaleBc::x_ex_func,this,zone_ptr->name+":x_ex");
          registerFunc(&RecycleRescaleBc::mean_u_func,this,zone_ptr->name+":mean_u_bc");
          registerFunc(&RecycleRescaleBc::p_bc_prof_func,this,zone_ptr->name+":p_bc");

          if ( mpi_rank == 0 )
            cout << " RecycleRescaleBc: " << param->str() << endl;

          zone_ptr->lb_cost = 200;
          exchanger         = NULL;
          dx                = NULL;

          int iarg = 1;
          while ( iarg < param->size() ) {
            string token = utils::toUpperCase(param->getString(iarg++));
            if ( token == "DX") {
              dx = new double[3];
              for (int i = 0; i < 3; ++i)
                dx[i] = param->getDouble(iarg++);
            } else if ( token == "MDOT") {
              mdot_bc = param->getDouble(iarg++);
            } else if ( token == "P") {
              p_bc    = param->getDouble(iarg++);
            } else if ( token == "RHO") {
              rho_bc  = param->getDouble(iarg++);
            } else if ( token == "LZ") { // optionally handle spanwise periodicity
              lz      = param->getDouble(iarg++); // assumes [0,Lz] domain
            } else if ( token == "DELTA_IN") {
              delta_in = param->getDouble(iarg++);
            } else if ( token == "LY") {
              ly      = param->getDouble(iarg++);
            } else if ( token == "RE_DELTA_IN") {
              re_delta_in = param->getDouble(iarg++);
            } else if ( token == "U_EDGE") {
              u_edge = param->getDouble(iarg++);
            } else if ( token == "Y_ZERO") {
              y_bc_zero = param->getDouble(iarg++);
            } else if ( token == "Y_ORIENT") {
              y_orientation = param->getDouble(iarg++);
            } else if ( (token == "DT_MEAN")||(token == "T_MEAN")) {
              t_mean  = param->getDouble(iarg++);
            } else if ( (token == "FILE")||(token == "FILENAME")||(token == "FILE_NAME")) {
              filename = param->getString(iarg++);
              COUT(" > FILE " << filename);
              // check that the file is available and readable...
              if (!utils::fileExists(filename)) {
                CERR("PROFILE FILE " << filename << " does not exist");
              }
              b_filename = true;
            } else if ( token == "FORMAT") {
              fileformat = param->getUCString(iarg++);
              if (mpi_rank == 0) cout << " > file format: " << fileformat << endl;
            } else {
              CERR( " > unrecognized token " << token <<
                    " during parse of bc:\n" << param->str());
            }
          }

          if ( dx == NULL) {
            CERR( " > RECYCLE_RESCALE_BC requires DX <dx0> <dx1> <dx2>");
          }

          if ( mdot_bc == BIG_DOUBLE) {
            CERR( " > RECYCLE_RESCALE_BC requires MDOT <mdot>");
          }

          if ( rho_bc == BIG_DOUBLE) {
            CERR( " > RECYCLE_RESCALE_BC requires RHO <rho-bc>");
          }

          if ( p_bc == BIG_DOUBLE) {
            CERR( " > RECYCLE_RESCALE_BC requires P <p-bc>");
          }
        
        }

        ~RecycleRescaleBc() {
        
          DELETE(dx);
          DELETE(x_ex);
          DELETE(u_send);
          DELETE(u_recv);
          
          DELETE(mean_u_ex);
          DELETE(mean_u_bc);

          DELETE(u_scale);
          DELETE(work_arr_host);
          DELETE(u_bc);
          DELETE(p_bc_prof);
          DELETE(h_bc_prof);

          if ( exchanger) {
            delete exchanger;
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( u_send_d) {
            cti::utils::freeOnDevice(u_send_d);
            u_send_d = NULL;
          }
          
          if ( u_recv_d) {
            cti::utils::freeOnDevice(u_recv_d);
            u_recv_d = NULL; 
          }

          if ( work_arr_d) {
            cti::utils::freeOnDevice(work_arr_d);
            work_arr_d = NULL;
          }

          if ( u_bc_d) {
            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;
          }

          if ( p_bc_prof_d) {
            cti::utils::freeOnDevice(p_bc_prof_d);
            p_bc_prof_d = NULL;
          }

          if ( h_bc_prof_d) {
            cti::utils::freeOnDevice(h_bc_prof_d);
            h_bc_prof_d = NULL;
          }

          if ( mean_u_ex_d) {
            cti::utils::freeOnDevice(mean_u_ex_d);
            mean_u_ex_d = NULL;
          }

          if ( mean_u_bc_d) {
            cti::utils::freeOnDevice(mean_u_bc_d);
            mean_u_bc_d = NULL;
          }

          if ( u_scale_d) {
            cti::utils::freeOnDevice(u_scale_d);
            u_scale_d = NULL;
          }

#endif
        
        }

        void initData() {
        
          if ( mpi_rank == 0) {
            cout << " > init exchanger for zone : " << zone_ptr->getName() << endl;
          }

          assert ( exchanger == NULL);

          assert( dx);
          assert( x_ex == NULL);
          x_ex = new double[zone_ptr->nbf][3];

          u_scale = new double[zone_ptr->nbf];

          assert(p_bc_prof == NULL);
          p_bc_prof = new double[zone_ptr->nbf];

          assert(h_bc_prof == NULL);
          h_bc_prof = new double[zone_ptr->nbf];

          // -------------------------------------------------------------
          // STEP 1: Read in the mean profile (same format as CbcProfile)
          // -------------------------------------------------------------

          assert( mean_u_bc == NULL);
          mean_u_bc = new double[zone_ptr->nbf][3];

          utils::ProfileReader profile;
          assert(b_filename);
          profile.init(filename,fileformat);
          profile.ensureVar("u");
          profile.ensureVar("v");
          profile.ensureVar("w");

          if (profile.getType() == utils::LINE_WD) {
            double * wall_dist = new double[zone_ptr->nbf];
            CtiData *dn = eval("distance(\"wall\")",true);
            double *dn_data = dn->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              wall_dist[ibf] = dn_data[zone_ptr->cvobf[ibf]];
            }
            profile.setPoints(wall_dist,zone_ptr->nbf);
            DELETE(wall_dist);
          }
          else if (profile.getType() == utils::POINT_3D) {
            profile.setPoints(zone_ptr->x_bf,zone_ptr->x_global,zone_ptr->nbf);
          }
          else {
            assert(profile.getType() == utils::LINE_R);
            CERR("LINE_R profile type currently not supported");
          }
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            mean_u_bc[ibf][0] = profile.getData(ibf,"u");
            mean_u_bc[ibf][1] = profile.getData(ibf,"v");
            mean_u_bc[ibf][2] = profile.getData(ibf,"w");
          }

          if (u_edge == BIG_DOUBLE) {
            double my_maxu = 0.0;
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              if (mean_u_bc[ibf][0] > my_maxu)
                my_maxu = mean_u_bc[ibf][0];
            }
            MPI_Reduce(&my_maxu,&u_edge,1,MPI_DOUBLE,MPI_MAX,0,mpi_comm);
            if ( mpi_rank == 0 )
              cout << " > RecycleRescaleBc: found u_edge = " << u_edge << endl;
          }

          // assert( mean_u_bc == NULL);
          // mean_u_bc = new double[zone_ptr->nbf][3];
          // assert(b_filename);
          // double *y_profile = NULL;
          // double *u_profile = NULL;
          // int n_profile;
          // int ierr = utils::xcol(y_profile,u_profile,n_profile,filename,1,2);
          // if (ierr != 0) {
          //   CERR("cannot parse columns 1 and 2 in profile file: " << filename);
          // }
          // for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          //   double y_bc = zone_ptr->x_bf[ibf][1];
          //   y_bc = ( y_bc - y_bc_zero ) * y_orientation; // adjust origin plane and normal direction
          //   if (y_bc <= y_profile[0]) {
          //     assert(0);
          //   }
          //   else if (y_bc >= y_profile[n_profile-1]) {
          //     mean_u_bc[ibf][0] = u_profile[n_profile-1];
          //     mean_u_bc[ibf][1] = 0.0;
          //     mean_u_bc[ibf][2] = 0.0;
          //   }
          //   else {
          //     // bisection...
          //     int left = 0;
          //     int right = n_profile-1;
          //     while ((right - left) > 1) {
          //       const int middle = (left + right)/2;   // equivalent to floor..
          //       if (y_bc >= y_profile[middle])
          //         left = middle;
          //       else
          //         right = middle;
          //     }
          //     const double w0 = (y_profile[right]-y_bc)/(y_profile[right]-y_profile[left]);
          //     mean_u_bc[ibf][0] = w0*u_profile[left] + (1.0-w0)*u_profile[right];
          //     mean_u_bc[ibf][1] = 0.0;
          //     mean_u_bc[ibf][2] = 0.0;
          //   }
          // }
          // delete[] y_profile;
          // delete[] u_profile;

          // -------------------------------------------------------------------
          // STEP 2: Choose the x_ex locations based on a turbulent correlation
          // -------------------------------------------------------------------

          // empirical correlation used in Urbin & Knight (2001)
          // delta_re/delta_in = [1+(dx/delta_in)*0.27^(6/5)*Re_deltain^(-1/5)]^(5/6)
          if ( delta_in != BIG_DOUBLE) {
            double dfac = 1.0 + (dx[0]/delta_in) * powf(0.27,1.2) * powf(re_delta_in,-0.2);
            dfac = powf(dfac,5.0/6.0);
            delta_re = delta_in * dfac;
          }

          // also used in Urbin & Knight (2001)
          // u_tau_in/u_tau_re = (delta_re/delta_in)^(1/10)
          double u_tau_fac = 1.0;
          if ( delta_in != BIG_DOUBLE) {
            u_tau_fac = powf((delta_re / delta_in),0.1);
            assert(u_tau_fac >= 1.0);
          }

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            FOR_I3 x_ex[ibf][i] = zone_ptr->x_bf[ibf][i] + dx[i];

            u_scale[ibf] = 1.0;

            // offset the recycling location for outer rescaling
            double dy_offset = 0.0;
            if ( delta_in != BIG_DOUBLE) {
              const double y_in = (zone_ptr->x_bf[ibf][1] - y_bc_zero) * y_orientation;
              if ( y_in <= delta_in ) {
                // inside BL
                dy_offset = (delta_re/delta_in)*y_in - y_in;
                dy_offset *= y_orientation;
                u_scale[ibf] = u_tau_fac;
              } else if ( y_in <= ly ) {
                // outside BL but rescaled
                dy_offset = (y_in - delta_in)*(ly - delta_re)/(ly - delta_in) + delta_re - y_in;
                dy_offset *= y_orientation;
                u_scale[ibf] = 0.0; // kill fluc here
              } else {
                // outside BL and not rescaled
                u_scale[ibf] = 0.0; // kill flux here
              }
            }
            x_ex[ibf][1] += dy_offset;

            if ( x_ex[ibf][2] == BIG_DOUBLE) continue;

            if ( lz == BIG_DOUBLE) continue;

            while ( x_ex[ibf][2] >= lz) { // handle spanwise periodicity
              x_ex[ibf][2] -= lz;
            }
            while ( x_ex[ibf][2] < 0) {
              x_ex[ibf][2] += lz;
            }
            assert(x_ex[ibf][2] >= 0 && x_ex[ibf][2] < lz);

          }

          exchanger = new DataExchanger();
          exchanger->init(x_ex,zone_ptr->nbf);

          // -----------------------------------------------------------------
          // STEP 3: Adjust points on periodic boundaries that were not found
          // -----------------------------------------------------------------

          // probably some points will not be "inside" domain
          // due to jagged periodic boundaries
          int my_adjust = 0;

          int * bf_flag = new int[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            bf_flag[ibf] = -1;

          for (int ii = 0; ii < exchanger->nx_unpack; ++ii)
            bf_flag[exchanger->idopt_recv[ii]]++;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            if (bf_flag[ibf] == -1) {
              // uh oh.. couldn't find this guy
              // maybe he's on the other side of the periodic boundary
              double x_old[3];
              FOR_I3 x_old[i] = x_ex[ibf][i];
              if (lz != BIG_DOUBLE) {
                if (x_ex[ibf][2] < lz/2) {
                  x_ex[ibf][2] += lz;
                } else {
                  x_ex[ibf][2] -= lz;
                }
              }
              my_adjust += 1;
              cout << " > adjusted: ibf " << ibf << " (" << x_old[0] << "," << x_old[1] << "," << x_old[2]
                   << ") -> (" << x_ex[ibf][0] << "," << x_ex[ibf][1] << "," << x_ex[ibf][2] << ")" << endl;
            }
          }

          int adjust;
          MPI_Reduce(&my_adjust,&adjust,1,MPI_INT,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0)
            cout << " > number of x_ex points adjusted : " << adjust << endl;

          delete exchanger; // make way for a new exchanger
          exchanger = NULL;
          delete[] bf_flag;
          bf_flag = NULL;

          exchanger = new DataExchanger();
          exchanger->init(x_ex,zone_ptr->nbf);

          // -----------------------------------------------------------------------
          // STEP 4: Adjust points on periodic boundaries that were found as ghosts
          // -----------------------------------------------------------------------

          if ( mpi_rank == 0) cout << " > now check for ghosts" << endl;

          // ... we need to make the data exchanger usable on cpu

          exchanger->buildSparseCommunicatorStruct();

          assert( u_send == NULL); u_send = new double[3*exchanger->nx_pack];
          assert( u_recv == NULL); u_recv = new double[3*exchanger->nx_unpack];

          //h_bc = p_bc/rho_bc*gamma/(gamma-1.0);

          assert(u_bc);
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            FOR_I3 u_bc[ibf][i] = 0.0;
          }

          // we need to do an exchanger to tell the recv ranks that the
          // send rank found a ghost cell so the recy rank can adjust
          // the exchange point accordingly

          int n_recv = exchanger->recv_stuff.size()/3;
          int n_send = exchanger->send_stuff.size()/3;

          MPI_Request * recv_req = new MPI_Request[n_recv];
          MPI_Request * send_req = new MPI_Request[n_send];

          int jj_recv = 0;
          int jj_send = 0;

          const int UPDATE_TAG = 11112;

          // post the receives first ...
          int my_recv_disp  = -1;
          int my_self_count = -1;

          for (int ii = 0; ii < n_recv; ++ii) {
          
            const int rank = exchanger->recv_stuff[3*ii];
            if ( rank != mpi_rank ) {
            
              MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                        3*exchanger->recv_stuff[3*ii+1],
                        MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                        &recv_req[jj_recv]);

              ++jj_recv;

            } else {
            
              // data sent to myself

              my_self_count = exchanger->recv_stuff[3*ii+1];
              my_recv_disp  = exchanger->recv_stuff[3*ii+2];
            
            }
          
          }

          // pack the data for the exchange ...

          for (int ii = 0; ii < exchanger->nx_pack; ++ii) {
            const int icv = exchanger->cvopt[ii];
            double val = 1.0;
            if ( !((icv >= 0)&&(icv < ncv)) ) {
              val = -1.0;
            }
            FOR_I3 u_send[3*ii+i] = val;
          }

          // now post the sends

          for (int ii = 0; ii < n_send; ++ii) {
          
            const int rank = exchanger->send_stuff[3*ii];

            if ( rank != mpi_rank) {
            
              MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2],
                         3*exchanger->send_stuff[3*ii+1],
                         MPI_DOUBLE,rank,
                         UPDATE_TAG,mpi_comm,&send_req[jj_send]);

              ++jj_send;
            
            } else {
            
              // check the counts for the self copy ...

              assert( exchanger->send_stuff[3*ii+1] == my_self_count);
            
            }
          
          }

          // the communication is in flight right now. while we wait,
          // if there are any self communications that present then
          // we can take care of them here

          if ( my_self_count > 0 ) {
          
            assert( n_send > 0);
            const int send_disp = exchanger->send_stuff[3*(n_send-1)+2];

            for (int ii = 0; ii < my_self_count; ++ii) {
              FOR_I3 u_recv[3*(my_recv_disp+ii)+i] = u_send[3*(send_disp+ii)+i];
            }
          
          }

          // now we have to wait ...
          MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

          // we copy everything that isnt the self comm. note
          // that the self comm on the recv side is also at the end...
          // all the data is now there.

          if ( exchanger->nx_unpack > 0) {
          
            for (int ii = 0; ii < exchanger->nx_unpack; ++ii) {
            
              const int ibf = exchanger->idopt_recv[ii];
              FOR_I3 u_bc[ibf][i] = u_recv[3*ii+i];
            
            }
          
          }

          // ensure the send complete .. and we're out

          MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

          delete[] send_req;
          delete[] recv_req;

          // check how many points are found in ghost cells
          int my_ghosts = 0;
          for (int ii = 0; ii < exchanger->nx_pack; ++ii) {
            const int icv = exchanger->cvopt[ii];
            if (!((icv >= 0)&&(icv < ncv))) {
              ++my_ghosts;
              cout << " > rank " << mpi_rank << " : ghost icv " << icv
                   << " x_cv (" << x_cv[icv][0] << "," << x_cv[icv][1] << "," << x_cv[icv][2] << ")"
                   << " x_ex (" << exchanger->x_pack[ii][0] << "," << exchanger->x_pack[ii][1] << "," << exchanger->x_pack[ii][2] << ")" << endl;
            }
          }
          int ghosts;
          MPI_Reduce(&my_ghosts,&ghosts,1,MPI_INT,MPI_SUM,0,mpi_comm);
          if (mpi_rank == 0)
            cout << " > number of ghost cells found : " << ghosts << endl;

          // now u_bc[ibf][i] > 0 for all i if the send rank found
          // a non-ghost cell that matches x_ex[ibf]              
          // and u_bc[ibf][i] < 0 for all i if the send rank found
          // a ghost cell that matches x_ex[ibf]
          // we need to adjust x_ex[ibf] if a ghost cell was found

          my_adjust = 0;

          if ( mpi_rank == 0)
            cout << " > Checking if DataExchanger retrieved ghost cells ..." << endl;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            if ( u_bc[ibf][0] < 0 ) {
              // uh oh.. this guy is a ghost on the send rank
              // let's try to find him on the other side
              double x_old[3];
              FOR_I3 x_old[i] = x_ex[ibf][i];
              //if ( lz != BIG_DOUBLE) {
              //  if (x_ex[ibf][2] < lz/2) {
              //    x_ex[ibf][2] += lz;
              //  } else {
              //    x_ex[ibf][2] -= lz;
              //  }
              //}
              my_adjust += 1;
              cout << " > adjusted: ibf " << ibf << " (" << x_old[0] << "," << x_old[1] << "," << x_old[2]
                   << ") -> (" << x_ex[ibf][0] << "," << x_ex[ibf][1] << "," << x_ex[ibf][2] << ")" << endl;
            }
          }

          adjust = 0;
          MPI_Reduce(&my_adjust,&adjust,1,MPI_INT,MPI_SUM,0,mpi_comm);

          if ( mpi_rank == 0)
            cout << " > number of x_ex points adjusted : " << adjust << endl;

          delete exchanger; // make way for a new exchanger
          exchanger = NULL;
          delete[] bf_flag;
          bf_flag = NULL;

          delete[] u_send; u_send = NULL;
          delete[] u_recv; u_recv = NULL;

          // ----------------------------------------------------------
          // STEP 5: Create the DataExchanger and check that it's okay
          // ----------------------------------------------------------

          exchanger = new DataExchanger();
          exchanger->init(x_ex,zone_ptr->nbf);

          // make sure all exchange points are inside domain...
          
          bf_flag = new int[zone_ptr->nbf];
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            bf_flag[ibf] = -1;

          for (int ii = 0; ii < exchanger->nx_unpack; ++ii)
            bf_flag[exchanger->idopt_recv[ii]]++;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            if (bf_flag[ibf] == -1) {
              cout << " > could not find exchange point: " << COUT_VEC(x_ex[ibf]) << ". Please modify DX." << endl;
              cout.flush();
              assert(0);
            }
          }

          delete[] bf_flag;

          // check how many points are found in ghost cells
          my_ghosts = 0;
          for (int ii = 0; ii < exchanger->nx_pack; ++ii) {
            const int icv = exchanger->cvopt[ii];
            if (!((icv >= 0)&&(icv < ncv))) {
              ++my_ghosts;
              cout << " > rank " << mpi_rank << " : ghost icv " << icv
                   << " x_cv ("<<x_cv[icv][0]<<","<<x_cv[icv][1]<<","<<x_cv[icv][2]<<")" 
                   << " x_ex ("<<exchanger->x_pack[ii][0]<<","<<exchanger->x_pack[ii][1]<<","<<exchanger->x_pack[ii][2]<<")" << endl;
            }
          }
          ghosts = 0;
          MPI_Reduce(&my_ghosts,&ghosts,1,MPI_INT,MPI_SUM,0,mpi_comm);
          if (mpi_rank == 0)
            cout << " > number of ghost cells found : " << ghosts << endl;

          // we expect that the communicator is sparse...
          
          exchanger->buildSparseCommunicatorStruct();
          n_recv = exchanger->recv_stuff.size()/3;

          {
          
            // actually lets check the worst case complexity of the
            // communicator ...

            int self_count = 0;
            for (int ii = 0; ii < n_recv; ++ii) {
            
              int rank = exchanger->recv_stuff[3*ii];
              if ( rank == mpi_rank) {
                // this is the self send ..
                self_count = exchanger->recv_stuff[3*ii+1];
                break;
              }
            }

            double my_fax = double(zone_ptr->nbf-self_count)/double(zone_ptr->nbf);
            double fax;
            MPI_Reduce(&my_fax,&fax,1,MPI_DOUBLE,MPI_MAX,0,mpi_comm);

            if ( mpi_rank == 0 )
              cout << " > worst case exch fraction : " << fax << endl;
          
          }
          
          // ----------------------------------------------------
          // STEP 6: Allocate remaining data
          // ----------------------------------------------------

          // lastly allocate some data for the vel magnitude ...

          assert( u_send == NULL); u_send = new double[3*exchanger->nx_pack];
          assert( u_recv == NULL); u_recv = new double[3*exchanger->nx_unpack];

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( u_send_d == NULL);
          assert( u_recv_d == NULL);

          cti::utils::initOnDevice(u_send_d,3*exchanger->nx_pack);
          cti::utils::initOnDevice(u_recv_d,3*exchanger->nx_unpack);

#endif

          doVelocityExchange();
          
          // setting initial mean_u_ex after doVelocityExchange() got u_bc
          if ( !checkInit(mean_u_ex,zone_ptr->nbf)) {
          
            if ( mpi_rank == 0)
              cout << " > resetting mean_u_ex for zone: " << zone_ptr->getName() << endl;

            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              for (int i = 0; i < 3; ++i)
                mean_u_ex[ibf][i] = u_bc[ibf][i];
            }
          
          }
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            for (int i = 0; i < 3; ++i) {
              u_bc[ibf][i] -= mean_u_ex[ibf][i];
              u_bc[ibf][i] += mean_u_bc[ibf][i];
            }
          }

          // setting pressure from Bernoulli
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            p_bc_prof[ibf] = p_bc - 0.5*rho_bc*MAG2(u_bc[ibf]);
            // compressible correction
            p_bc_prof[ibf] = p_bc*pow(1.0+0.5*(gamma-1.0)*rho_bc*MAG2(u_bc[ibf])/gamma/p_bc_prof[ibf],-gamma/(gamma-1.0));
            p_bc_prof[ibf] = p_bc*pow(1.0+0.5*(gamma-1.0)*rho_bc*MAG2(u_bc[ibf])/gamma/p_bc_prof[ibf],-gamma/(gamma-1.0));
            h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);

            const double y_in = (zone_ptr->x_bf[ibf][1] - y_bc_zero) * y_orientation;
            if ( y_in <= delta_in ) {
              // correct the total pressure inside BL to edge pressure
              p_bc_prof[ibf] = p_bc - 0.5*rho_bc*u_edge*u_edge;
              // compressible correction
              p_bc_prof[ibf] = p_bc*pow(1.0+0.5*(gamma-1.0)*rho_bc*u_edge*u_edge/gamma/p_bc_prof[ibf],-gamma/(gamma-1.0));
              p_bc_prof[ibf] = p_bc*pow(1.0+0.5*(gamma-1.0)*rho_bc*u_edge*u_edge/gamma/p_bc_prof[ibf],-gamma/(gamma-1.0));
              h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);
            }

          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert( u_bc_d == NULL);
          cti::utils::initAndCopyToDevice(u_bc_d,(double*)u_bc,3*zone_ptr->nbf);
          
          cti::utils::initAndCopyToDevice(p_bc_prof_d, p_bc_prof, zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(h_bc_prof_d, h_bc_prof, zone_ptr->nbf);

          cti::utils::initAndCopyToDevice(mean_u_ex_d,(double*)mean_u_ex,3*zone_ptr->nbf);
          
          assert( mean_u_bc_d == NULL);
          cti::utils::initAndCopyToDevice(mean_u_bc_d,(double*)mean_u_bc,3*zone_ptr->nbf);

          assert( u_scale_d == NULL);
          cti::utils::initAndCopyToDevice(u_scale_d,u_scale,zone_ptr->nbf);

          assert( work_arr_d == NULL);
          cti::utils::initOnDevice(work_arr_d,1024);

          assert( work_arr_host == NULL);
          work_arr_host = new double[1024];

#endif

          if ( mpi_rank == 0)
            cout << " > " << zone_ptr->getName() << " completed initialization" << endl;
        
        }

        CtiData *x_ex_func(vector<CtiData*>& argVec,const bool b_eval) {
          // -----------------------------------------------------
          // x_ex() returns the coordinate component of the
          // exchange point for each ibf
          // -----------------------------------------------------
          int comp = -1;
          if (argVec.size() != 1) {
            eval_cerr << "x_ex() takes 1 argument: got " << argVec.size();
            return NULL;
          }
          if (argVec[0]->getDatatype() != I_DATA) {
            eval_cerr << "x_ex() argument expects I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
            return NULL;
          }
          comp = argVec[0]->i();
          if ( (comp < 0) || (comp >= 3) ) {
            eval_cerr << "x_ex() integer argument should be in [0:3), got " << comp << endl;
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) {
            double *rval_dn = rval->dn();
            assert( (comp >= 0)&&(comp < 3) );
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              rval_dn[ibf] = x_ex[ibf][comp];
            }
          }
          return rval;
        }

        CtiData *p_bc_prof_func(vector<CtiData*>& argVec,const bool b_eval) {
          if (argVec.size() != 0) {
            eval_cerr << "p_bc() takes no arguments: got " << argVec.size();
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) {
            double *rval_dn = rval->dn();
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              rval_dn[ibf] = p_bc_prof[ibf];
            }
          }
          return rval;
        }

        CtiData *mean_u_func(vector<CtiData*>& argVec,const bool b_eval) {
          int comp = -1;
          if (argVec.size() != 1) {
            eval_cerr << "mean_u_bc() takes 1 argument: got " << argVec.size();
            return NULL;
          }
          if (argVec[0]->getDatatype() != I_DATA) {
            eval_cerr << "mean_u_bc() argument expects I_DATA, got " << datatypeAsString(argVec[0]->getDatatype()) << endl;
            return NULL;
          }
          comp = argVec[0]->i();
          if ( (comp < 0) || (comp >= 3) ) {
            eval_cerr << "mean_u_bc() integer argument should be in [0:3), got " << comp << endl;
            return NULL;
          }

          CtiData *rval = new CtiData(DN_DATA,zone_ptr->nbf,b_eval);
          if (b_eval) {
            double *rval_dn = rval->dn();
            assert( (comp >= 0)&&(comp < 3) );
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              rval_dn[ibf] = mean_u_bc[ibf][comp];
            }
          }
          return rval;
        }

        void preLoadBalance() {
        
          DELETE(u_bc);
          DELETE(p_bc_prof);
          DELETE(h_bc_prof);
          DELETE(u_send);
          DELETE(u_recv);
          DELETE(work_arr_host);

          if ( exchanger) {
            delete exchanger;
            exchanger = NULL;
          }

#if defined(BUILD_CUDA) || defined(BUILD_HIP) 

          if ( u_bc_d ) { 
            cti::utils::freeOnDevice(u_bc_d);
            u_bc_d = NULL;
          } 

          if ( p_bc_prof_d ) {
            cti::utils::freeOnDevice(p_bc_prof_d);
            p_bc_prof_d = NULL;
          }

          if ( h_bc_prof_d ) {
            cti::utils::freeOnDevice(h_bc_prof_d);
            h_bc_prof_d = NULL;
          }

          if ( u_send_d ) { 
            cti::utils::freeOnDevice(u_send_d);
            u_send_d = NULL;
          } 

          if ( u_recv_d ) { 
            cti::utils::freeOnDevice(u_recv_d);
            u_recv_d = NULL;
          }

          if ( mean_u_ex_d) {
            cti::utils::freeOnDevice(mean_u_ex_d);
            mean_u_ex_d = NULL;
          }

          if ( mean_u_bc_d) {
            cti::utils::freeOnDevice(mean_u_bc_d);
            mean_u_bc_d = NULL;
          }

          if ( u_scale_d) {
            cti::utils::freeOnDevice(u_scale_d);
            u_scale_d = NULL;
          }

          if (work_arr_d) { 
            cti::utils::freeOnDevice(work_arr_d);
            work_arr_d = NULL;
          } 

#endif
        
        }

        void postLoadBalance() {
          initData();
        }

        void doVelocityExchange() {
        
          assert( exchanger);
          const int n_recv = exchanger->recv_stuff.size()/3;
          const int n_send = exchanger->send_stuff.size()/3;

          MPI_Request * recv_req = new MPI_Request[n_recv];
          MPI_Request * send_req = new MPI_Request[n_send];

          int jj_recv = 0;
          int jj_send = 0;

          const int UPDATE_TAG = 11112;

          // post the receives first ...

          int my_recv_disp  = -1;
          int my_self_count = -1;

          for (int ii = 0; ii < n_recv; ++ii) {
          
            const int rank = exchanger->recv_stuff[3*ii];
            if ( rank != mpi_rank ) {
            
              MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                        3*exchanger->recv_stuff[3*ii+1],
                        MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                        &recv_req[jj_recv]);

              ++jj_recv;

            } else {
            
              // this is data i am sending to myself, so record
              // the displacement

              my_self_count = exchanger->recv_stuff[3*ii+1];
              my_recv_disp  = exchanger->recv_stuff[3*ii+2];
            
            }
          
          }

          // pack the data for the exchange ...

          for (int ii = 0; ii < exchanger->nx_pack; ++ii) {
            const int icv = exchanger->cvopt[ii];
            if (!((icv >= 0)&&(icv < ncv_g2))) {
              cout << " >>> rank " << mpi_rank << " : "
                   << " icv " << icv << " ncv " << ncv
                   << " ncv_g2 " << ncv_g2 << endl;
            }
            assert((icv >= 0)&&(icv < ncv_g2));
            FOR_I3 u_send[3*ii+i] = u[icv][i];
          }

          // now post the sends

          for (int ii = 0; ii < n_send; ++ii) {
          
            const int rank = exchanger->send_stuff[3*ii];

            if ( rank != mpi_rank) {
            
              MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2],
                         3*exchanger->send_stuff[3*ii+1],
                         MPI_DOUBLE,rank,
                         UPDATE_TAG,mpi_comm,&send_req[jj_send]);

              ++jj_send;
            
            } else {
            
              // check the counts for the self copy ...

              assert( exchanger->send_stuff[3*ii+1] == my_self_count);
            
            }
          
          }

          // the communication is in flight right now. while we wait,
          // if there are any self communications that present then
          // we can take care of them here

          if ( my_self_count > 0 ) {
          
            assert( n_send > 0);
            const int send_disp = exchanger->send_stuff[3*(n_send-1)+2];

            for (int ii = 0; ii < my_self_count; ++ii) {
              FOR_I3 u_recv[3*(my_recv_disp+ii)+i] = u_send[3*(send_disp+ii)+i];
            }
          
          }

          // now we have to wait ...

          MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

          // we copy everything that isnt the self comm. note
          // that the self comm on the recv side is also at the end...
          // all the data is now there.

          if ( exchanger->nx_unpack > 0 ) {
          
            for (int ii = 0; ii < exchanger->nx_unpack; ++ii) {
            
              const int ibf = exchanger->idopt_recv[ii];
              FOR_I3 u_bc[ibf][i] = u_recv[3*ii+i];
            
            }
          
          }

          // ensure the sends complete .. and we're out.

          MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

          delete[] send_req;
          delete[] recv_req;

          // // calc mf global...

          // double my_buf = 0.0;
          // for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          //   const int icv = zone_ptr->cvobf[ibf];
          //   my_buf += rho[icv]*DOT_PRODUCT(u_bc[ibf],zone_ptr->n_bf[ibf]);
          // }
          // double buf;
          // MPI_Allreduce(&my_buf,&buf,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

          // // rescale...

          // assert(buf != 0.0);
          // buf = -mdot_bc/buf;
          // for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          //   FOR_I3 u_bc[ibf][i] *= buf;
          // }
        
        }

        void doVelocityExchangeGpu() {
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          assert(exchanger);
          const int n_recv = exchanger->recv_stuff.size()/3;
          const int n_send = exchanger->send_stuff.size()/3;

          MPI_Request * recv_req = new MPI_Request[n_recv];
          MPI_Request * send_req = new MPI_Request[n_send];

          int jj_recv = 0;
          int jj_send = 0;

          const int UPDATE_TAG = 11112;

          // post the receives first ...

          int my_recv_disp  = -1;
          int my_self_count = -1;

          for (int ii = 0; ii < n_recv; ++ii) {
          
            const int rank = exchanger->recv_stuff[3*ii];
            if ( rank != mpi_rank ) {
            
              MPI_Irecv(u_recv+3*exchanger->recv_stuff[3*ii+2],
                        3*exchanger->recv_stuff[3*ii+1],
                        MPI_DOUBLE,rank,UPDATE_TAG,mpi_comm,
                        &recv_req[jj_recv]);

              ++jj_recv;
            
            } else {
            
              // this is data i am sending to myself, so record
              // the displacement
              
              my_self_count = exchanger->recv_stuff[3*ii+1];
              my_recv_disp  = exchanger->recv_stuff[3*ii+2];
            
            }

          }
          
          // pack the data for the exchange ...

          if ( exchanger->nx_pack > 0) {
          
            const int block_size = 128;
            const int grid_size  = (exchanger->nx_pack + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL(packVelocityForExchange,
                              grid_size, block_size,
                              u_send_d, u_f_d, exchanger->cvopt_d, exchanger->nx_pack);

          }

          // XXX should consider using pinned memory for this transfer ...
          // i do not need to copy the whole send buffer bc some of the 
          // data is going to be local to this rank

          int n_pack = 0;
          if ( n_send > 0 ) {
          
            if ( exchanger->send_stuff[3*(n_send-1)] == mpi_rank ) {
            
              // the send_disp is the count of the pack that isnt
              // related to the self send ...
              
              n_pack = exchanger->send_stuff[3*(n_send-1)+2];
            
            } else {
            
              // i dont have a self send, so i need to whole buffer
              
              n_pack = exchanger->nx_pack;
            
            }
          
          }

          if ( n_pack > 0 ) {
            cti::utils::copyToHost(u_send,u_send_d,3*n_pack);
          }

          // now post the sends
          
          for (int ii = 0; ii < n_send; ++ii) {
          
            const int rank = exchanger->send_stuff[3*ii];

            if ( rank != mpi_rank) {
            
              MPI_Issend(u_send+3*exchanger->send_stuff[3*ii+2],
                         3*exchanger->send_stuff[3*ii+1],
                         MPI_DOUBLE,rank,
                         UPDATE_TAG,mpi_comm,&send_req[jj_send]);

              ++jj_send;
            
            } else {
            
              // check the counts for the self copy ...

              assert( exchanger->send_stuff[3*ii+1] == my_self_count);
            
            }
          
          }

          // the communication is in flight right now. while we wait,
          // if there are any self communications that present then
          // we can take care of them here
          
          if ( my_self_count > 0 ) {
          
            assert( n_send > 0);
            const int send_disp = exchanger->send_stuff[3*(n_send-1)+2];
            const int block_size = 128;
            const int grid_size = (my_self_count + block_size - 1)/block_size;

            LAUNCH_GPU_KERNEL(performExchangeSelfComm,
                              grid_size, block_size,
                              u_recv_d, u_send_d, send_disp,
                              my_recv_disp, my_self_count);

          }

          // now we have to wait ...

          MPI_Waitall(jj_recv,recv_req,MPI_STATUSES_IGNORE);

          // we copy everything that isnt the self comm. note
          // that the self comm on the recv side is also at the end...

          int n_unpack = 0;
          if ( n_recv > 0 ) {
          
            if ( exchanger->recv_stuff[3*(n_recv-1)] == mpi_rank ) {
            
              // the send_disp is the count of the pack that isnt
              // related to the self send ...

              n_unpack = exchanger->recv_stuff[3*(n_recv-1)+2];
            
            } else {
            
              // i dont have a self send, so i need to whole buffer

              n_unpack = exchanger->nx_unpack;
            
            }
          
          }

          if ( n_unpack > 0 ) {
            cti::utils::copyToDevice(u_recv_d,u_recv, 3*n_unpack);
          }

          // all the data is now there.
          
          if ( exchanger->nx_unpack > 0 ) {
          
            const int block_size = 128;
            const int grid_size  = (exchanger->nx_unpack + block_size - 1)/block_size;

            LAUNCH_GPU_KERNEL(unpackVelocityForExchange,
                              grid_size, block_size,
                              u_bc_d, u_recv_d, exchanger->idopt_recv_d, exchanger->nx_unpack);
          
          }

          // ensure the sends complete ... and we're out.
          
          MPI_Waitall(jj_send,send_req,MPI_STATUSES_IGNORE);

          delete[] send_req;
          delete[] recv_req;

          // // calc mf global...

          // if (zone_ptr->nbf > 0) {
          // 
          //   {
          //   
          //     const int block_size = 128;
          //     const int grid_size  = 128; // must be smaller than max block size...

          //     LAUNCH_GPU_KERNEL(mf_bc_reduce_block2, grid_size, block_size,
          //                       work_arr_d, rho_d, u_bc_d, zone_ptr->n_bf_d,
          //                       zone_ptr->cvobf_d, zone_ptr->nbf);
          //   
          //   }

          //   {
          //   
          //     // reduce in place ...
          //     
          //     const int block_size = 128; // must equal grid_size above..
          //     const int grid_size  =   1;

          //     cti::utils::reduce(block_size,block_size,grid_size,
          //                        work_arr_d,work_arr_d);

          //     // we need to copy back the value in red_work_arr[0] to the host
          //     // and then pass through the MPI reduce
          //     cti::utils::copyToHost(work_arr_host,work_arr_d,1);

          //   }
          // 
          // }
          // else {
          //   work_arr_host[0] = 0.0;
          // }

          // double tmp;
          // MPI_Allreduce(work_arr_host,&tmp,1,MPI_DOUBLE,MPI_SUM,mpi_comm);

          // if ( zone_ptr->nbf > 0) {
          //   
          //   assert(tmp != 0.0);
          //   tmp = -mdot_bc/tmp;

          //   {
          //   
          //     const int block_size = 256;
          //     const int grid_size  = (3*zone_ptr->nbf + block_size - 1) / block_size;

          //     LAUNCH_GPU_KERNEL( sax, grid_size, block_size,
          //                        u_bc_d, tmp, 0, 3*zone_ptr->nbf);
          //   
          //   }

          // }

#endif
        
        }

        void updateExchangeMean() {
          // u_bc contains the raw exchange velocity
          // mean_u_bc contains the mean profile for the bc
          //
          // compute mean_u_ex using exponential time filter

          const double eps_l = dt/(dt+t_mean);

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
            FOR_I3 {
              mean_u_ex[ibf][i] *= (1.0-eps_l);
              mean_u_ex[ibf][i] += eps_l*u_bc[ibf][i];
              u_bc[ibf][i] -= mean_u_ex[ibf][i];
              u_bc[ibf][i] *= u_scale[ibf];
              u_bc[ibf][i] += mean_u_bc[ibf][i];
            }

            //p_bc_prof[ibf] = p_bc - 0.5*rho_bc*MAG2(u_bc[ibf]);
            //h_bc_prof[ibf] = p_bc_prof[ibf]/rho_bc*gamma/(gamma-1.0);
          }
        
        }
        
        void updateExchangeMeanGpu() {
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0) {
          
            const double eps_l = dt/(dt+t_mean);

            const int block_size = 128;
            const int grid_size  = (3*zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL(updateExchangeMeanDevice,
                              grid_size, block_size,
                              u_bc_d, mean_u_ex_d, mean_u_bc_d,
                              u_scale_d, p_bc_prof_d, h_bc_prof_d,
                              (cti_ffp) p_bc, (cti_ffp) rho_bc,
                              (cti_ffp) gamma,
                              eps_l, zone_ptr->nbf);

          }

#endif
        
        }
        
        void addBoundaryFlux(double (*rhs)[5],const int rk_stage) {
        
          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          
            double flux[5];
            const int icv = zone_ptr->cvobf[ibf];
            calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                            rho[icv],u[icv],p[icv],h[icv],gamma,
                            rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc_prof[ibf],gamma);
            // mass flux needs to be stored for potential passive scalars..
            //mf[ibf] = flux[0];
            FOR_I5 rhs[icv][i] -= flux[i];
          
          }
        
        }

        void addBoundaryFluxGpu( cti_ffp * rhs, const int rk_stage,
                                 gpuStream_t& stream) {
        
#if defined(BUILD_CUDA) || defined(BUILD_HIP)

          if ( zone_ptr->nbf > 0 ) {
          
            const int block_size = 128;
            const int grid_size  = (zone_ptr->nbf + block_size - 1) / block_size;

            LAUNCH_GPU_KERNEL_STREAM( addCbcProfIgBcFlux, grid_size,
                                      block_size, stream, rhs,
                                      sp_vol_f_d, u_f_d, p_f_d, h_f_d,
                                      (cti_ffp)rho_bc, u_bc_d, p_bc_prof_d,
                                      h_bc_prof_d, zone_ptr->cvobf_d,
                                      zone_ptr->n_bf_d, zone_ptr->nbf,
                                      (cti_ffp)gamma);
          
          }

#endif
        
        }

        void rkStep(const double *rk_wgt,const int rk_stage) {
        
          if ( rk_stage == 1) {
            doVelocityExchange();
            updateExchangeMean();
          } 
        
        }

        void rkStepGpu(const double * rk_wgt, const int rk_stage) {
        
          if ( rk_stage == 1) {
            doVelocityExchangeGpu();
            updateExchangeMeanGpu();
          }

        }

        void syncToHost() {
#if defined(BUILD_CUDA) || defined(BUILD_HIP)
          cti::utils::copyToHost((double*)u_bc,u_bc_d,3*zone_ptr->nbf);
#endif
        }

        void query() {
        
          // grab the static pressure, temperature and the inlet mass flux ...

          double my_buf[4];

          for (int i = 0; i < 4; ++i)
            my_buf[i] = 0.0;

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
          
            const int icv = zone_ptr->cvobf[ibf];
            my_buf[0]    += zone_ptr->area_bf[ibf];
            my_buf[1]    += zone_ptr->area_bf[ibf] * p[icv];
            my_buf[2]    += zone_ptr->area_bf[ibf] * T[icv];
            my_buf[3]    += rho[icv] * DOT_PRODUCT( u[icv], zone_ptr->n_bf[ibf]);

          }

          double buf[4];
          MPI_Reduce(my_buf, buf, 4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          if ( mpi_rank == 0 ) {
          
            cout << "QUERY_BC " << zone_ptr->getName()
                 << " 4:step " << step
                 << " 6:time " << time
                 << " 8:p " << buf[1]/buf[0]
                 << " 10:T " << buf[2]/buf[0]
                 << " 12:mdot " << buf[3]
                 << endl;
          
          }
        
        }

        void force(double (*rval_dn3)[3], const ForceType f_type) {
        
          // TODO gpu

          for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf)
            FOR_I3 rval_dn3[ibf][i] = 0.0;

          if (f_type==ForceType::TOTAL_FORCE) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc_prof[ibf],gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p_ref*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::CONVECTIVE_FORCE) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv0 = zone_ptr->cvobf[ibf];
              double flux[5];
              calcRiemannFlux(flux,zone_ptr->n_bf[ibf],
                              rho[icv0],u[icv0],p[icv0],h[icv0],gamma,
                              rho_bc,u_bc[ibf],p_bc_prof[ibf],h_bc_prof[ibf],gamma);
              FOR_I3 rval_dn3[ibf][i] += flux[1+i] - p[icv0]*zone_ptr->n_bf[ibf][i];
            }
          }
          else if (f_type==ForceType::PRESSURE_FORCE) {
            for (int ibf = 0; ibf < zone_ptr->nbf; ++ibf) {
              const int icv0 = zone_ptr->cvobf[ibf];
              FOR_I3 rval_dn3[ibf][i] += (p[icv0]-p_ref)*zone_ptr->n_bf[ibf][i];
            }
          }
        }

      };


      void parseBcs() {

	COUT("parseBcs()");
	assert(bcVec.empty());
	bcVec.resize(bfZoneVec.size());
	for (int izn = 0, nzn = bfZoneVec.size(); izn < nzn; ++izn) {
	  // note that not all bcs will be set here if the user is using 
	  // custom bcs. These will get parsed in registerHook...
	  bcVec[izn] = NULL;
	  const string zone_name = bfZoneVec[izn].getName();
	  if (Param *param = getParamNoUsage(zone_name)) {
	    const string bc_type = param->getUCString(0);
            if ((bc_type == "SLIP")||(bc_type == "SYMMETRY")) {
	      bcVec[izn] = new SlipBc(&bfZoneVec[izn],param);
	    }
	    else if (bc_type == "WALL_ADIABATIC") {
	      bcVec[izn] = new WallAdiabaticBc(&bfZoneVec[izn],param);
	    }
            else if ( bc_type == "WALL_ISOTHERMAL") { 
              bcVec[izn] = new WallIsothermalBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "WALL_HEAT_FLUX") { 
              bcVec[izn] = new WallHeatFluxBc(&bfZoneVec[izn],param);
            }
	    else if (bc_type == "WM_ALG_ADIABATIC") {
	      bcVec[izn] = new WmAlgAdiabaticBc(&bfZoneVec[izn],param);
	    }
            else if ( bc_type == "WM_ALG_EXCHANGE") { 
              bcVec[izn] = new WmExchangeAdiabaticBc(&bfZoneVec[izn],param);
            }
	    else if (bc_type == "WM_ALG_ISOTHERMAL") {
	      bcVec[izn] = new WmAlgIsothermalBc(&bfZoneVec[izn],param);
	    }
	    else if (bc_type == "WM_ALG_CHT") {
	      bcVec[izn] = new WmAlgChtBc(&bfZoneVec[izn],param);
	      b_cht_bcs = true;
	    }
	    else if (bc_type == "WM_SLIP_ISOTHERMAL_V") {
	      bcVec[izn] = new WmSlipIsothermalVBc(&bfZoneVec[izn],param);
	    }
            else if (bc_type == "WM_SLIP_ISOTHERMAL") {
	      bcVec[izn] = new WmSlipIsothermalBc(&bfZoneVec[izn],param);
	    }
            else if ( bc_type == "WM_SLIP_ISOTHERMAL_HYBRID") { 
              bcVec[izn] = new WmSlipIsothermalHybridBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "WM_OPTIMAL_SLIP_ISOTHERMAL") {
              bcVec[izn] = new WmOptSlipBc(&bfZoneVec[izn],param);
            }
	    else if (bc_type == "WMI_ALG_ADIABATIC") {
	      bcVec[izn] = new WmiAlgAdiabaticBc(&bfZoneVec[izn],param);
	    }
            else if (bc_type == "WM_ROUGH_KES_ADIABATIC") {
              bcVec[izn] = new WmRoughKESAdiabaticBc(&bfZoneVec[izn],param);
            }
	    else if (bc_type == "CBC_RUP") {
	      bcVec[izn] = new CbcRupBc(&bfZoneVec[izn],param);
	    }
	    else if (bc_type == "CBC_UPT") {
	      bcVec[izn] = new CbcUptBc(&bfZoneVec[izn],param);
	    }
	    else if (bc_type == "CBC_MPT") {
	      bcVec[izn] = new CbcMptBc(&bfZoneVec[izn],param);
	    }
	    else if (bc_type == "CBC_PROFILE") {
	      bcVec[izn] = new CbcProfileBc(&bfZoneVec[izn],param);
	    }
            else if (bc_type == "CBC_ZERO_VORT") {
              bcVec[izn] = new CbcZeroVortBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "NSCBC_MT") { 
              bcVec[izn] = new NscbcMtBc(&bfZoneVec[izn],param);
            }
            else if ( (bc_type == "NSCBC_OUTLET_PRESSURE") || (bc_type == "NSCBC_OUTLET_P")) { 
              bcVec[izn] = new NscbcOutletPressureBc(&bfZoneVec[izn],param);
            }
	    else if ( bc_type == "NSCBC_OUTLET_PRESSURE_PROFILE") { 
              bcVec[izn] = new NscbcOutletPressureProfileBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "NSCBC_OUTLET_MDOT") { 
              bcVec[izn] = new NscbcOutletMdotBc(&bfZoneVec[izn],param);
            } 
            else if ( bc_type == "NSCBC_OUTLET_MDOT_UPDATE") { 
              bcVec[izn] = new NscbcOutletMdotUpdateBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "RECYCLE_BC") { 
              bcVec[izn] = new RecycleBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "RECYCLE_RESCALE_BC") {
              bcVec[izn] = new RecycleRescaleBc(&bfZoneVec[izn],param);
            }
            else if ( bc_type == "CBC_TOTAL_PT") { 
              bcVec[izn] = new CbcTotalPtBc(&bfZoneVec[izn],param);
            }
	    else if ( bc_type == "CBC_TOTAL_PT_PROFILE") {
	      bcVec[izn] = new CbcTotalPtProfileBc(&bfZoneVec[izn],param);
	    }
	    else {
	      if (mpi_rank == 0) {
		cout << " > zone " << zone_name << " has unrecognized bc_type: " << bc_type << " (may be custom?)" << endl;
	      }
	    }
	  }
	}
      
      }

    }
  }
}
