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
 * @brief Contact Detection for Quadrupeds based on force/torque or pressure, and encoder measurement
 * @author Stylianos Piperakis
 * @details Estimates the feet contact state and the gait-phase
 */

#include <serow/Gaussian.h>
#include <string.h>
#include <serow/mediator.h>

namespace serow
{
class ContactDetectionQuad
{
private:
  ///instance for Gaussian distribution operations
  Gaussian gs;
  ///uncertainty in left front / left hind/ right front / left hind GRF
  double sigmaLFf, sigmaLHf, sigmaRFf, sigmaRHf;
  ///uncertainty in left front / left hind/ right front / left hind COP
  double sigmaLFc, sigmaLHc, sigmaRFc, sigmaRHc;
  ///uncertainty in left front / left hind/ right front / left hind foot velocity
  double sigmaLFv, sigmaLHv, sigmaRFv, sigmaRHv;
  ///min/max foot polygon length/width (assuming identical left front/ left hind/right front/right hind foot)
  double xmin, xmax, ymin, ymax;
  ///min/max contact force
  double LFfmin, LHfmin, RFfmin, RHfmin;
  ///probability of left front / left hind/ right front / left hind contact based on a) force, b) cop c) foot velocity
  double pLFf, pLHf, pRFf, pRHf, pLFc, pLHc, pRFc, pRHc, pLFv, pLHv, pRFv, pRHv;
  ///probability of contact for left front / left hind/ right front / left hind leg
  double pLF, pLH, pRF, pRH, p;
  ///minimum foot velocity threshold
  double VelocityThres;
  ///flags for first detected contact/consider COP in estimation/consider kinematics
  bool firstContact, useCOP, useKin;
  ///instantenous indicator of left front / left hind/ right front / left hind leg contact - filtered with a median filter
  int contactLF, contactLH, contactRF, contactRH, sum, contactRFFilt, contactRHFilt, contactLFFilt, contactLHFilt;
  ///probabistic threshold on contact
  double prob_TH;
  ///support foot frame name, support leg name, left front foot frame name, left hind foot frame name, right front foot frame name, right hind foot frame name, gait-phase
  std::string support_foot_frame, support_leg, LFfoot_frame, LHfoot_frame, RFfoot_frame, RHfoot_frame, phase;
  ///median filter buffer size 
  int medianWindow;
  ///median filters for left front, left hind, right front, right hind,foot
	Mediator *LFmdf, *LHmdf, *RFmdf, *RHmdf;
  ///mass of the robot and gravity constant
  double mass, g;


  double deltaLFfz, deltaLHfz, deltaRFfz, deltaRHfz, LFf_, LHf_, RFf_, RHf_;
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
  /** @fn double  computeContactProb(double LFf, double LHf, double RFf, double RHf,
                            double copLFx, double copLFy, double copLHx,
                            double copLHy, double copRFx, double copRFy,
                            double copRHx, double copRHy, double LFvnorm,
                            double LHvnorm, double RFvnorm, double RHvnorm)
    *   @brief computes the contact probability assuming a Gaussian distribution on the measured COP/force/foot velocity
    *   @param LFf  left front foot vertical force
    *   @param LHf  left hind foot vertical force
    *   @param RFf  right front foot vertical force
    *   @param RHf  right hind foot vertical force
    *   @param copLFx  cop in left front foot - x axis
    *   @param copLFy  cop in left front foot - y axis
    *   @param copLHx  cop in left hind foot - x axis
    *   @param copLHy  cop in left hind foot - y axis
    *   @param copRFx  cop in right front foot - x axis
    *   @param copRFy  cop in right front foot - y axis
    *   @param copRHx  cop in right hind foot - x axis
    *   @param copRHy  cop in right hind foot - y axis
    *   @param LFvnorm  left foot front velocity norm
    *   @param LHvnorm  left foot hind velocity norm
    *   @param RFvnorm  right foot front velocity norm
    *   @param RHvnorm  right foot hind velocity norm
  */
  double computeContactProb(double LFf, double LHf, double RFf, double RHf,
                            double copLFx, double copLFy, double copLHx,
                            double copLHy, double copRFx, double copRFy,
                            double copRHx, double copRHy, double LFvnorm,
                            double LHvnorm, double RFvnorm, double RHvnorm)
  {

    pLFf = computeForceContactProb(LFfmin, sigmaLFf, LFf);
    pLF = pLFf;

    pLHf = computeForceContactProb(LHfmin, sigmaLHf, LHf);
    pLH = pLHf;

    pRFf = computeForceContactProb(RFfmin, sigmaRFf, RFf);
    pRF = pRFf;

    pRHf = computeForceContactProb(RHfmin, sigmaRHf, RHf);
    pRH = pRHf;

 

    //To utilize the full contact-wrench - careful for non-coplanar contacts
    if (useCOP)
    {
      pLFc = computeCOPContactProb(xmax, xmin, sigmaLFc, copLFx) * computeCOPContactProb(ymax, ymin, sigmaLFc, copLFy);
      pLHc = computeCOPContactProb(xmax, xmin, sigmaLHc, copLHx) * computeCOPContactProb(ymax, ymin, sigmaLHc, copLHy);

      pRFc = computeCOPContactProb(xmax, xmin, sigmaRFc, copRFx) * computeCOPContactProb(ymax, ymin, sigmaRFc, copRFy);
      pRHc = computeCOPContactProb(xmax, xmin, sigmaRHc, copRHx) * computeCOPContactProb(ymax, ymin, sigmaRHc, copRHy);

      pLF *= pLFc;
      pRF *= pRFc;
      pLH *= pLHc;
      pRH *= pRHc;
    }

    if (useKin)
    {
      pLFv = computeKinContactProb(VelocityThres, sigmaLFv, LFvnorm);
      pLHv = computeKinContactProb(VelocityThres, sigmaLHv, LHvnorm);

      pRFv = computeKinContactProb(VelocityThres, sigmaRFv, RFvnorm);
      pRHv = computeKinContactProb(VelocityThres, sigmaRHv, RHvnorm);

      pLF *= pLFv;
      pLH *= pLHv;

      pRF *= pRFv;
      pRH *= pRHv;
    }

    p = pLF + pLH + pRF + pRH;

    if (p != 0)
    {
      pLF = pLF / p;
      pRF = pRF / p;
      pLH = pLH / p;
      pRH = pRH / p;
    }
    else
    {
      pLF = 0;
      pRF = 0;
      pLH = 0;
      pRH = 0;
    }

    contactLH = 0;
    if (pLF >= prob_TH)
      contactLF = 1;

    contactRF = 0;
    if (pRF >= prob_TH)
      contactRF = 1;

    contactLH = 0;
    if (pLH >= prob_TH)
      contactLH = 1;

    contactRH = 0;
    if (pRH >= prob_TH)
      contactRH = 1;




  }

public:
  void init(std::string LFfoot_frame_, std::string LHfoot_frame_, std::string RFfoot_frame_, std::string RHfoot_frame_, double LFfmin_, double LHfmin_, double RFfmin_, double RHfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmaLFf_, double sigmaLHf_, double sigmaRFf_, double sigmaRHf_, double sigmaLFc_, double sigmaLHc_, double sigmaRFc_, double sigmaRHc_, double VelocityThres_,
            double sigmaLFv_, double sigmaLHv_, double sigmaRFv_, double sigmaRHv_, bool useCOP_ = false, bool useKin_ = false, double prob_TH_ = 0.9, int medianWindow_=  10)
  {
    LFfoot_frame = LFfoot_frame_;
    LHfoot_frame = LHfoot_frame_;

    RFfoot_frame = RFfoot_frame_;
    RHfoot_frame = RHfoot_frame_;

    LFfmin = LFfmin_;
    LHfmin = LHfmin_;
    RFfmin = RFfmin_;
    RHfmin = RHfmin_;

    xmin = xmin_;
    xmax = xmax_;
    ymin = ymin_;
    ymax = ymax_;

    sigmaLFf = sigmaLFf_;
    sigmaLHf = sigmaLHf_;
    sigmaRFf = sigmaRFf_;
    sigmaRHf = sigmaRHf_;

    sigmaLFc = sigmaLFc_;
    sigmaLHc = sigmaLHc_;
    sigmaRFc = sigmaRFc_;
    sigmaRHc = sigmaRHc_;

    sigmaLFv = sigmaLFv_;
    sigmaLHv = sigmaLHv_;

    sigmaRFv = sigmaRFv_;
    sigmaRHv = sigmaRHv_;

    contactLF = 0;
    contactLH = 0;

    contactRF = 0;
    contactRH = 0;

    pLF = 0;
    pLH = 0;

    pRF = 0;
    pRH = 0;

    pLFf = 0;
    pLHf = 0;

    pRFf = 0;
    pRHf = 0;

    pLFc = 0;
    pLHc = 0;

    pRFc = 0;
    pRHc = 0;

    pLFv = 0;
    pLHv = 0;

    pRFv = 0;
    pRHv = 0;

    VelocityThres = VelocityThres_;
    prob_TH = prob_TH_;
    useCOP = useCOP_;
    useKin = useKin_;
    firstContact = true;

    deltaLFfz = 0;
    deltaLHfz = 0;

    deltaRFfz = 0;
    deltaRHfz = 0;
  }

  void init(std::string LFfoot_frame_, std::string LHfoot_frame_, std::string RFfoot_frame_, std::string RHfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_, double prob_TH_ = 0.9, double mass_ = 1, double g_=9.81, int medianWindow_=  10)
  {

    LFfoot_frame = LFfoot_frame_;
    LHfoot_frame = LHfoot_frame_;

    RFfoot_frame = RFfoot_frame_;
    RHfoot_frame = RHfoot_frame_;

    LegHighThres = LegHighThres_;
    LegLowThres = LegLowThres_;

    StrikingContact = StrikingContact_;
    VelocityThres = VelocityThres_;

    firstContact = true;
    prob_TH = prob_TH_;
    pLF = 0;
    pLH = 0;

    pRF = 0;
    pRH = 0;
    mass = mass_;
    g = g_;
    contactLF = 0;
    contactLH = 0;
    contactRF = 0;
    contactRH = 0;
    contactLFFilt = 0;
    contactRFFilt = 0;
    contactRHFilt = 0;
    contactLHFilt = 0;



    deltaLFfz = 0;
    deltaLHfz = 0;

    deltaRFfz = 0;
    deltaRHfz = 0;
    medianWindow = medianWindow_;
    LFmdf = MediatorNew(medianWindow);
    LHmdf = MediatorNew(medianWindow);
    RFmdf = MediatorNew(medianWindow);
    RHmdf = MediatorNew(medianWindow);
    cout<<"Gait-Phase Estimation Module Initialized"<<endl;
  }
  double getLFDiffForce()
  {
    return fabs(deltaLFfz);
  }

  double getRFDiffForce()
  {
    return fabs(deltaRFfz);
  }
  double getLHDiffForce()
  {
    return fabs(deltaLHfz);
  }
  double getRHDiffForce()
  {
    return fabs(deltaRHfz);
  }
  void computeSupportFoot(double LFf, double LHf, double RFf, double RHf,
                          double copLFx, double copLFy, double copLHx, double copLHy,
                          double copRFx, double copRFy, double copRHx, double copRHy,
                          double LFvnorm, double LHvnorm, double RFvnorm, double RHvnorm)
  {

    computeContactProb(LFf, LHf, RFf, RHf, copLFx, copLFy, copLHx, copLHy, copRFx, copRFy, copRHx, copRHy, LFvnorm, LHvnorm, RFvnorm, RHvnorm);
    if (pLF > pLH && pLF > pRF && pLF > pRH)
    {
      // Initial support leg
      support_leg = "LFLeg";
      support_foot_frame = LFfoot_frame;
    }
    else if (pLH > pLF && pLH > pRF && pLH > pRH)
    {
      support_leg = "LHLeg";
      support_foot_frame = LHfoot_frame;
    }
    else if (pRF > pLF && pRF > pLH && pRF > pRH)
    {
      support_leg = "RFLeg";
      support_foot_frame = RFfoot_frame;
    }
    else
    {
      support_leg = "RHLeg";
      support_foot_frame = RHfoot_frame;
    }

    sum = contactRH + contactRF + contactLF + contactRH;

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
    case 3:
      phase = "Triple Support";
    case 4:
      phase = "Quad Support";
    }

    if (firstContact)
    {
      deltaLFfz = 0;
      deltaRFfz = 0;
      deltaLHfz = 0;
      deltaRHfz = 0;
      firstContact = false;
    }
    else
    {
      deltaLFfz = LFf - LFf_;
      deltaRFfz = RFf - RFf_;
      deltaLHfz = LHf - LHf_;
      deltaRHfz = RHf - RHf_;
    }

    LFf_ = LFf;
    RFf_ = RFf;
    LHf_ = LHf;
    RHf_ = RHf;
  }

  double getLFLegContactProb()
  {
    return pLF;
  }
  double getLHLegContactProb()
  {
    return pLH;
  }
  double getRFLegContactProb()
  {
    return pRF;
  }
  double getRHLegContactProb()
  {
    return pRH;
  }
  int isLFLegContact()
  {
    return contactLFFilt;
  }

  int isRFLegContact()
  {
    return contactRFFilt;
  }

  int isLHLegContact()
  {
    return contactLHFilt;
  }

  int isRHLegContact()
  {
    return contactRHFilt;
  }

  std::string getSupportFrame()
  {
    return support_foot_frame;
  }

  std::string getSupportPhase()
  {
    return phase;
  }
  std::string getSupportLeg()
  {
    return support_leg;
  }

  void SchmittTriggerWithKinematics(double LFf, double LHf, double RFf, double RHf, double LFvnorm, double LHvnorm, double RFvnorm, double RHvnorm)
  {

    if (LFf > LHf && LFf > RFf && LFf > RHf)
    {
      support_leg = "LFLeg";
      support_foot_frame = LFfoot_frame;
    }
    else if (LHf > LFf && LHf > RFf && LHf > RHf)
    {
      support_leg = "LHLeg";
      support_foot_frame = LHfoot_frame;
    }
    else if (RFf > LFf && RFf > LHf && RFf > RHf)
    {
      support_leg = "RFLeg";
      support_foot_frame = RFfoot_frame;
    }
    else
    {
      support_leg = "RHLeg";
      support_foot_frame = RHfoot_frame;
    }

    if (LFvnorm <= VelocityThres)
    {
      if (contactLF == 0)
      {
        if (LFf > LegHighThres && LFf < StrikingContact)
        {
          contactLF = 1;
        }
      }
      else
      {
        if (LFf < LegLowThres)
        {
          contactLF = 0;
        }
      }
    }

    if (LHvnorm <= VelocityThres)
    {
      if (contactLH == 0)
      {
        if (LHf > LegHighThres && LHf < StrikingContact)
        {
          contactLH = 1;
        }
      }
      else
      {
        if (LHf < LegLowThres)
        {
          contactLH = 0;
        }
      }
    }

    if (RFvnorm <= VelocityThres)
    {
      if (contactRF == 0)
      {
        if (RFf > LegHighThres && RFf < StrikingContact)
        {
          contactRF = 1;
        }
      }
      else
      {
        if (RFf < LegLowThres)
        {
          contactRF = 0;
        }
      }
    }

    if (RHvnorm <= VelocityThres)
    {
      if (contactRH == 0)
      {
        if (RHf > LegHighThres && RHf < StrikingContact)
        {
          contactRH = 1;
        }
      }
      else
      {
        if (RHf < LegLowThres)
        {
          contactRH = 0;
        }
      }
    }

    sum = contactRH + contactRF + contactLF + contactRH;

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
    case 3:
      phase = "Triple Support";
    case 4:
      phase = "Quad Support";
    }

    if (!firstContact)
    {
      deltaLFfz = LFf - LFf_;
      deltaRFfz = RFf - RFf_;
      deltaLHfz = LHf - LHf_;
      deltaRHfz = RHf - RHf_;
    }
    else
    {
      deltaLFfz = 0;
      deltaRFfz = 0;
      deltaLHfz = 0;
      deltaRHfz = 0;
      firstContact = false;
    }

    LFf_ = LFf;
    RFf_ = RFf;
    LHf_ = LHf;
    RHf_ = RHf;
  }
  void computeForceWeights(double LFf, double LHf, double RFf, double RHf)
  {
    p = LFf + LHf + RFf + RHf;

    LFf = cropGRF(LFf);
    LHf = cropGRF(LHf);
    RFf = cropGRF(RFf);
    RHf = cropGRF(RHf);

    pLF = 0;
    pRF = 0;
    pLH = 0;
    pRH = 0;

    if (p > 0)
    {
      pLF = LFf / p;
      pRF = RFf / p;
      pLH = LHf / p;
      pRH = RHf / p;
    }

     pLF=normalizeProb(pLF);
     pRF=normalizeProb(pRF);
     pLH=normalizeProb(pLH);
     pRH=normalizeProb(pRH);
  }




  double cropGRF(double f_)
  {
      return max(0.0, min(f_, mass * g));
  }

  double normalizeProb(double s_)
  {
    double res = s_;
    if(s_<0)
      res=0;

    if(s_>1)
      res=1;
    
    return res;
  }

  void SchmittTrigger(double LFf, double LHf, double RFf, double RHf)
  {



    if (LFf >= LHf && LFf >= RFf && LFf >= RHf)
    {
      support_leg = "LFLeg";
      support_foot_frame = LFfoot_frame;
    }
    else if (LHf >= LFf && LHf >= RFf && LHf >= RHf)
    {
      support_leg = "LHLeg";
      support_foot_frame = LHfoot_frame;
    }
    else if (RFf >= LFf && RFf >= LHf && RFf >= RHf)
    {
      support_leg = "RFLeg";
      support_foot_frame = RFfoot_frame;
    }
    else
    {
      support_leg = "RHLeg";
      support_foot_frame = RHfoot_frame;
    }

    if (contactLF == 0)
    {
      if (LFf > LegHighThres && LFf < StrikingContact)
      {
        contactLF = 1;
      }
    }
    else
    {
      if (LFf < LegLowThres)
      {
        contactLF = 0;
      }
    }

    if (contactLH == 0)
    {
      if (LHf > LegHighThres && LHf < StrikingContact)
      {
        contactLH = 1;
      }
    }
    else
    {
      if (LHf < LegLowThres)
      {
        contactLH = 0;
      }
    }

    if (contactRF == 0)
    {
      if (RFf > LegHighThres && RFf < StrikingContact)
      {
        contactRF = 1;
      }
    }
    else
    {
      if (RFf < LegLowThres)
      {
        contactRF = 0;
      }
    }

    if (contactRH == 0)
    {
      if (RHf > LegHighThres && RHf < StrikingContact)
      {
        contactRH = 1;
      }
    }
    else
    {
      if (RHf < LegLowThres)
      {
        contactRH = 0;
      }
    }


    MediatorInsert(LFmdf, contactLF);
    contactLFFilt = MediatorMedian(LFmdf);
    MediatorInsert(LHmdf, contactLH);
    contactLHFilt = MediatorMedian(LHmdf);
    MediatorInsert(RFmdf, contactRF);
    contactRFFilt = MediatorMedian(RFmdf);
    MediatorInsert(RHmdf, contactRH);
    contactRHFilt =  MediatorMedian(RHmdf);

    // cout<<"Contact -- LF -- LH -- RF -- RH"<<endl;
    // cout<<pLF<<endl;
    // cout<<pLH<<endl;
    // cout<<pRF<<endl;
    // cout<<pRH<<endl;

    sum = contactRHFilt + contactRFFilt + contactLFFilt + contactRHFilt;

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
    case 3:
      phase = "Triple Support";
    case 4:
      phase = "Quad Support";
    }

    if (!firstContact)
    {
      deltaLFfz = LFf - LFf_;
      deltaRFfz = RFf - RFf_;
      deltaLHfz = LHf - LHf_;
      deltaRHfz = RHf - RHf_;
    }
    else
    {
      deltaLFfz = 0;
      deltaRFfz = 0;
      deltaLHfz = 0;
      deltaRHfz = 0;
      firstContact = false;
    }

    LFf_ = LFf;
    RFf_ = RFf;
    LHf_ = LHf;
    RHf_ = RHf;
  }
};
} // namespace serow
