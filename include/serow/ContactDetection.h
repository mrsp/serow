/* 
 * Copyright 2017-2020 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *		 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @brief Contact Detection for Bipeds based on force/torque or pressure, and encoder measurement
 * @author Stylianos Piperakis
 * @details Estimates the feet contact state and the gait-phase
 */
#include <serow/Gaussian.h>
#include <string.h>




namespace serow
{
class ContactDetection
{
private:
  ///instance for Gaussian distribution operations
  Gaussian gs;
  ///uncertainty in left/right GRF
  double sigmalf, sigmarf;
  ///uncertainty in left/right COP
  double sigmalc, sigmarc;
  ///uncertainty in left/right foot velocity
  double sigmalv, sigmarv;
  ///min/max foot polygon length/width (assuming identical left/right foot)
  double xmin, xmax, ymin, ymax;
  ///min/max contact force
  double lfmin, rfmin;
  ///probability of left/right contact based on a) force, b) cop c) foot velocity
  double plf, prf, plc, prc, plv, prv;
  ///probability of contact for left/right leg
  double pl, pr, p;
  ///minimum foot velocity threshold
  double VelocityThres;
  ///flags for first detected contact/consider COP in estimation/consider kinematics
  bool firstContact, useCOP, useKin;
  ///instantenous indicator of left/right leg contact - filtered with a median filter
  int contactL, contactR, contactRFilt, contactLFilt, sum;
  ///probabistic threshold on contact
  double prob_TH;
  ///support foot frame name, support leg name, left foot frame name, right foot frame name, gait-phase
  std::string support_foot_frame, support_leg, lfoot_frame, rfoot_frame, phase;
  ///median filter buffer size for contactLFilt/contactRFilt
  int medianWindow;
  ///median filters for left and right foot
	Mediator *lmdf, *rmdf;
  double deltalfz, deltarfz, lf_, rf_;
  ///Smidtt Trigger detection thresholds
  double LegLowThres, LegHighThres, StrikingContact;

  /** @fn double computeForceContactProb(double fmin, double sigma, double f)
    *   @brief computes the contact probability assuming a Gaussian distribution on the measured force
    *   @param fmin min contact force
    *   @param sigma measured force uncertainty
    *   @param f measured force
  */
  double computeForceContactProb(double fmin, double sigma, double f)
  {
    return 1.000 - gs.cdf(fmin, f, sigma);
  } 

  /** @fn double computeKinContactProb(double vel_min, double sigma, double v)
    *   @brief computes the contact probability assuming a Gaussian distribution on the foot velocity
    *   @param vel_min min velocity 
    *   @param sigma velocity uncertainty
    *   @param v velocity
  */
  double computeKinContactProb(double vel_min, double sigma, double v)
  {
    return gs.cdf(vel_min, v, sigma);
  }

  /** @fn double computeCOPContactProb(double max, double min, double sigma, double cop)
    *   @brief computes the contact probability assuming a Gaussian distribution on the measured COP
    *   @param max max foot polygon length
    *   @param min min foot polygon length
    *   @param sigma uncertaintiy in COP
    *   @param cop measured cop
  */
  double computeCOPContactProb(double max, double min, double sigma, double cop)
  {
    if (cop != 0)
    {
      return gs.cdf(max, cop, sigma) - gs.cdf(min, cop, sigma);
    }
    else
    {
      return 0;
    }
  }

  /** @fn double computeContactProb(double lf, double rf, double coplx, double coply, double coprx, double copry, double lv, double rv)
    *   @brief computes the contact probability assuming a Gaussian distribution on the measured COP/force/foot velocity
    *   @param lf  left foot vertical force
    *   @param rf  right foot vertical force
    *   @param coplx  cop in left foot - x axis
    *   @param coply  cop in left foot - y axis
    *   @param coprx  cop in right foot - x axis
    *   @param copry  cop in right foot - y axis
    *   @param lv  left foot velocity norm
    *   @param rv  right foot velocity norm
  */
  double computeContactProb(double lf, double rf, double coplx, double coply, double coprx, double copry, double lv, double rv)
  {

    plf = computeForceContactProb(lfmin, sigmalf, lf);
    pl = plf;

    prf = computeForceContactProb(rfmin, sigmarf, rf);
    pr = prf;

    //To utilize the full contact-wrench - careful for non-coplanar contacts
    if (useCOP)
    {
      plc = computeCOPContactProb(xmax, xmin, sigmalc, coplx) * computeCOPContactProb(ymax, ymin, sigmalc, coply);
      prc = computeCOPContactProb(xmax, xmin, sigmarc, coprx) * computeCOPContactProb(ymax, ymin, sigmarc, copry);
      pl *= plc;
      pr *= prc;
    }
    if (useKin)
    {
      plv = computeKinContactProb(VelocityThres, sigmalv, lv);
      prv = computeKinContactProb(VelocityThres, sigmarv, rv);
      pl *= plv;
      pr *= prv;
    }

    p = pl + pr;

    if (p > 0)
    {
      pl = pl / p;
      pr = pr / p;
    }
    else
    {
      pl = 0;
      pr = 0;
    }

    contactL = 0;
    if (pl >= prob_TH)
      contactL = 1;

    contactR = 0;
    if (pr >= prob_TH)
      contactR = 1;

    MediatorInsert(lmdf, contactL);
    contactLFilt = MediatorMedian(lmdf); 
    MediatorInsert(rmdf, contactR);
    contactRFilt = MediatorMedian(rmdf);

    sum = contactRFilt + contactLFilt;

    switch(sum)
    {
    case 0:
      phase = "None";
      break;
    case 1:
      phase = "Single Support";
      break;
    case 2:
      phase = "Double Support";
      break;
    }

  }

public:
  /** @fn init(std::string lfoot_frame_, std::string rfoot_frame_, double lfmin_, double rfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmalf_, double sigmarf_, double sigmalc_, double sigmarc_, double VelocityThres_, double sigmalv_, double sigmarv_, bool useCOP_ = false, bool useKin_ = false, double prob_TH_ = 0.9 , int medianWindow_=  10)
  *   @brief initializes the Gait-phase Estimation Module for probabilistic contact estimation
  */
  void init(std::string lfoot_frame_, std::string rfoot_frame_, double lfmin_, double rfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmalf_, double sigmarf_, double sigmalc_, double sigmarc_, double VelocityThres_, double sigmalv_, double sigmarv_, bool useCOP_ = false, bool useKin_ = false, double prob_TH_ = 0.9 , int medianWindow_=  10)
  {
    lfoot_frame = lfoot_frame_;
    rfoot_frame = rfoot_frame_;
    lfmin = lfmin_;
    rfmin = rfmin_;
    xmin = xmin_;
    xmax = xmax_;
    ymin = ymin_;
    ymax = ymax_;
    sigmalf = sigmalf_;
    sigmarf = sigmarf_;
    sigmalc = sigmalc_;
    sigmarc = sigmarc_;
    sigmalv = sigmalv_;
    sigmarv = sigmarv_;
    contactL = 0;
    contactR = 0;
    contactRFilt = 0;
    contactLFilt = 0;
    pl = 0;
    pr = 0;
    plf = 0;
    prf = 0;
    plc = 0;
    prc = 0;
    plv = 0;
    prv = 0;
    VelocityThres = VelocityThres_;
    prob_TH = prob_TH_/2.0;
    useCOP = useCOP_;
    useKin = useKin_;
    firstContact = true;
    deltalfz = 0;
    deltarfz = 0;
    medianWindow = medianWindow_;
    rmdf = MediatorNew(medianWindow);
    lmdf = MediatorNew(medianWindow);

    cout<<"Gait-Phase Estimation Module Initialized"<<endl;
  }
  
  
  /** @fn init(std::string lfoot_frame_, std::string rfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_, double prob_TH_ = 0.9, int medianWindow_=  10)
  *   @brief initializes the Gait-phase Estimation Module for  contact estimation with a Schmitt-Trigger
  */
  void init(std::string lfoot_frame_, std::string rfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_, double prob_TH_ = 0.9, int medianWindow_=  10)
  {

    lfoot_frame = lfoot_frame_;
    rfoot_frame = rfoot_frame_;
    LegHighThres = LegHighThres_;
    LegLowThres = LegLowThres_;
    StrikingContact = StrikingContact_;
    VelocityThres = VelocityThres_;
    firstContact = true;
    prob_TH = prob_TH_;
    pl = 0;
    pr = 0;
    deltalfz = 0;
    deltarfz = 0;
    contactL = 0;
    contactR = 0;
    contactRFilt = 0;
    contactLFilt = 0;
    medianWindow = medianWindow_;
    rmdf = MediatorNew(medianWindow);
    lmdf = MediatorNew(medianWindow);
    cout<<"Gait-Phase Estimation Module with Schmitt-Trigger Initialized"<<endl;
  }
  /** @fn getDiffForce()
   *  @brief returns the absolute vertical force difference between the two legs
   */
  double getDiffForce()
  {
    cout<<"Delta FORCE"<<endl;
    cout<<fabs(deltalfz + deltarfz)<<endl;
    return fabs(deltalfz + deltarfz);
  }
  /** @fn computeSupportFoot(double lf, double rf, double coplx, double coply, double coprx, double copry, double lvnorm, double rvnorm)
   *  @brief computes both the contact probabilities and identifies the support foot
   */
  void computeSupportFoot(double lf, double rf, double coplx, double coply, double coprx, double copry, double lvnorm, double rvnorm)
  {

   
    computeContactProb(lf, rf, coplx, coply, coprx, copry, lvnorm, rvnorm);
    //Choose Initial Support Foot based on Contact Force
      if (contactLFilt ==1 && contactRFilt == 0)
      {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else if (contactRFilt ==1 && contactLFilt == 0)
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      else if (contactRFilt == 1 && contactLFilt == 1)
      {
        if(pl>pr)
        {
          support_leg = "LLeg";
          support_foot_frame = lfoot_frame;
        }
        else
        {
         support_leg = "RLeg";
         support_foot_frame = rfoot_frame;
        }
        
      }
      else
      {
        support_leg = "None";
         support_foot_frame = "None";
      }
    

    if(!firstContact)
    {
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    else
    {
      firstContact = false; 
    }
    
    lf_ = lf;
    rf_ = rf;
  }
  /** @fn double getLLegContactProb()
   *  @brief  returns the contact probability for the left foot
   */
  double getLLegContactProb()
  {
    return pl;
  }
  /** @fn double getRLegContactProb()
   *  @brief  returns the contact probability for the right foot
   */
  double getRLegContactProb()
  {
    return pr;
  }
  /** @fn  int isLLegContact()
   *  @brief  checks if the left foot is in contact
   */
  int isLLegContact()
  {
    return contactLFilt;
  }
  /** @fn  int isRLegContact()
   *  @brief  checks if the right foot is in contact
   */
  int isRLegContact()
  {
    return contactRFilt;
  }
  /** @fn   std::string getSupportFrame()
   *  @brief  returns the support foot frame
   */
  std::string getSupportFrame()
  {
    return support_foot_frame;
  }
  /** @fn   std::string getSupportLeg()
   *  @brief  returns the support leg
   */
  std::string getSupportLeg()
  {
    return support_leg;
  }
  /** @fn   std::string getSupportPhase()
   *  @brief  returns the gait phase
  */
  std::string getSupportPhase()
  {
    return phase;
  }
  /** @fn   computeForceWeights(double lf, double rf)
   *  @brief  returns weights for each leg based on measured forces
   *  @param lf left foot force
   *  @param rf right foot force
  */
  void computeForceWeights(double lf, double rf)
  {
    p = lf + rf;
    pl = 0;
    pr = 0;

    if (p > 0)
    {
      pl = lf / p;
      pr = rf / p;
    }
    if(pl < 0)
      pl = 0;
    
    if(pl > 1.0)
      pl = 1.0;

    if(pr < 0)
      pr = 0;

    if(pr > 1.0)
      pr = 1.0;



  }
  /** @fn  SchmittTrigger(double lf, double rf)
   *  @brief  applies a digital Schmitt-Trigger for contact detection
   *  @param lf left foot force
   *  @param rf right foot force
  */
  void SchmittTrigger(double lf, double rf)
  {
  



    if (contactL == 0)
    {
      if (lf > LegHighThres )
      {
        contactL = 1;
      }
    }
    else
    {
      if (lf < LegLowThres)
      {
        contactL = 0;
      }
    }

    if (contactR == 0)
    {
      if (rf > LegHighThres)
      {
        contactR = 1;
      }
    }
    else
    {
      if (rf < LegLowThres)
      {
        contactR = 0;
      }
    }

    MediatorInsert(lmdf, contactL);
    contactLFilt = MediatorMedian(lmdf); 
    MediatorInsert(rmdf, contactR);
    contactRFilt = MediatorMedian(rmdf);

    sum = contactRFilt + contactLFilt;

    switch(sum)
    {
    case 0:
      phase = "None";
      break;
    case 1:
      phase = "Single Support";
      break;
    case 2:
      phase = "Double Support";
      break;
    }

    if (contactLFilt == 1 && contactRFilt == 1)
    {
      if(pl>pr){
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      
    }
    else if(contactLFilt == 1)
    {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
    }
    else if(contactRFilt == 1)
    {
      support_leg = "RLeg";
      support_foot_frame = rfoot_frame;
    }
    else
    {
      support_leg = "None";
      support_foot_frame = "None";
    }
    
    if (!firstContact)
    {
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    else
    {
      firstContact = false;
    }

    lf_ = lf;
    rf_ = rf;
  }
 };
}