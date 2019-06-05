#include <serow/Gaussian.h>
#include <string.h>



namespace serow
{
class ContactDetectionQuad
{
private:
  Gaussian gs;
  //Probabilistic Kinematic - Contact Wrench detection
  double sigmaLFf, sigmaLHf, sigmaRFf, sigmaRHf;
  double sigmaLFc, sigmaLHc, sigmaRFc, sigmaRHc;
  double sigmaLFv, sigmaLHv, sigmaRFv, sigmaRHv;
  double xmin, xmax, ymin, ymax;
  double LFfmin, LHfmin, RFfmin, RHfmin;
  double pLFf,pLHf, pRFf, pRHf, pLFc, pLHc, pRFc, pRHc, pLFv, pLHv, pRFv, pRHv;
  double pLF, pLH, pRF, pRH, p;
  double VelocityThres;
  bool firstContact, useCOP, useKin;
  int contactLF, contactLH, contactRF, contactRH;
  double prob_TH;
  std::string support_foot_frame, support_leg, LFfoot_frame, LHfoot_frame, RFfoot_frame, RHfoot_frame;

  double deltaLFfz, deltaLHfz, deltaRFfz, deltaRHfz, LFf_, LHf_, RFf_, RHf_;
  //Smidtt Trigger detection
  double LegLowThres, LegHighThres, StrikingContact;

  double computeForceContactProb(double fmin, double sigma, double f)
  {
    return 1.000 - gs.cdf(fmin, f, sigma);
  }

  double computeKinContactProb(double vel_min, double sigma, double v)
  {
    return gs.cdf(vel_min, v, sigma);
  }


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

  double computeContactProb(double LFf,double LHf,double RFf,double RHf,
                            double copLFx,double copLFy,double  copLHx,
                            double copLHy,double copRFx,double copRFy,
                            double copRHx,double copRHy,double LFvnorm,
                            double LHvnorm,double RFvnorm,double RHvnorm)
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
      pRHv = computeKinContactProb(VelocityThres, sigmaRFv, RHvnorm);

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
      pLF += 0.5;
      pRF += 0.5;
      pLH += 0.5;
      pRH += 0.5;
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
  void init(std::string LFfoot_frame_, std::string  LHfoot_frame_, std::string RFfoot_frame_, std::string  RHfoot_frame_, double LFfmin_, double LHfmin_, double RFfmin_, double RHfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmaLFf_, double sigmaLHf_, double sigmaRFf_, double sigmaRHf_, double sigmaLFc_, double sigmaLHc_, double sigmaRFc_,  double sigmaRHc_, double VelocityThres_, 
            double sigmaLFv_, double sigmaLHv_, double sigmaRFv_, double sigmaRHv_, bool useCOP_ = false, bool useKin_ = false, double prob_TH_ = 0.9)
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

  void init(std::string LFfoot_frame_, std::string LHfoot_frame_, std::string RFfoot_frame_, std::string RHfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_, double prob_TH_ = 0.9)
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

    contactLF = 0;
    contactLH = 0;

    contactRF = 0;
    contactRH = 0;

    deltaLFfz = 0;
    deltaLHfz = 0;

    deltaRFfz = 0;
    deltaRHfz = 0;

  }
  double getDiffForce()
  {
    return fabs(deltaLFfz + deltaRFfz + deltaLHfz + deltaRHfz);
  }
  void computeSupportFoot(double LFf, double LHf, double RFf, double RHf, 
                          double copLFx, double copLFy, double copLHx, double copLHy,
                          double copRFx, double copRFy, double copRHx, double copRHy,
                          double LFvnorm, double LHvnorm, double RFvnorm, double RHvnorm)
  {

   
    computeContactProb(LFf, LHf, RFf, RHf, copLFx, copLFy,  copLHx, copLHy, copRFx, copRFy, copRHx, copRHy, LFvnorm, LHvnorm, RFvnorm, RHvnorm);
    //Choose Initial Support Foot based on Contact Force
    if (firstContact)
    {
      if (pLF >= pLH && pLF >= pRF && pLF >= pRH)
      {
        // Initial support leg
        support_leg = "LFLeg";
        support_foot_frame = LFfoot_frame;
      }
      else if(pLH >= pLF && pLH >= pRF && pLH >= pRH)
      {
        support_leg = "LHLeg";
        support_foot_frame = LHfoot_frame;
      }
      else if(pRF >= pLF && pRF >= pLH && pRF >= pRH)
      {
        support_leg = "RFLeg";
        support_foot_frame = RFfoot_frame;
      }
      else
      {
        support_leg = "RHLeg";
        support_foot_frame = RHfoot_frame;
      }
      firstContact = false;
    }
    else
    {
      if ((pLF >= pLH && pLF >= pRF && pLF >= pRH) && (LFvnorm < VelocityThres && LHvnorm > VelocityThres && RFvnorm > VelocityThres && RHvnorm > VelocityThres))
      {
        support_leg = "LFLeg";
        support_foot_frame = LFfoot_frame;
      }
      else if ((pLH >= pLF && pLH >= pRF && pLH >= pRH) && (LHvnorm < VelocityThres && LFvnorm > VelocityThres && RFvnorm > VelocityThres && RHvnorm > VelocityThres))
      {
        support_leg = "LHLeg";
        support_foot_frame = LHfoot_frame;
      }
      else if (pRF >= pLF && pRF >= pLH && pRF >= pRH && (RFvnorm < VelocityThres && LFvnorm > VelocityThres && LHvnorm > VelocityThres && RHvnorm > VelocityThres))
      {
        support_leg = "RFLeg";
        support_foot_frame = RFfoot_frame;
      }
      else if(pRH >= pLF && pRH >= pLH && pRH >= pRF && (RHvnorm < VelocityThres && LFvnorm > VelocityThres && LHvnorm > VelocityThres && RFvnorm > VelocityThres))
      {
        support_leg = "RHLeg";
        support_foot_frame = RHfoot_frame;
      }
    }

    if(!firstContact)
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
    return contactLF;
  }

  int isRFLegContact()
  {
    return contactRF;
  }

  int isLHLegContact()
  {
    return contactLH;
  }

  int isRHLegContact()
  {
    return contactRH;
  }

  std::string getSupportFrame()
  {
    return support_foot_frame;
  }
  std::string getSupportLeg()
  {
    return support_leg;
  }

  void SchmittTriggerWithKinematics(double LFf, double LHf, double RFf, double RHf, double LFvnorm, double LHvnorm, double RFvnorm, double RHvnorm)
  {
    contactLF = 0;
    contactLH = 0;

    contactRF = 0;
    contactRH = 0;

    p = LFf + LFf + RFf + RHf;
    pLF = 0;
    pRF = 0;
    pLH = 0;
    pRH = 0;

    if (p != 0)
    {
      pLF = LFf / p + 0.5;
      pRF = RFf / p + 0.5;
      pLH = LHf / p + 0.5;
      pRH = RHf / p + 0.5;
    }

    if (firstContact)
    {
      if (LFf >= LHf && LFf >= RFf && LFf >= RHf)
      {
        // Initial support leg
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

      if (pLF >= prob_TH)
        contactLF = 1;

      contactRF = 0;
      if (pRF >= prob_TH)
        contactRF = 1;

      if (pLH >= prob_TH)
        contactLH = 1;

      contactRH = 0;
      if (pRH >= prob_TH)
        contactRH = 1;

      firstContact = false;
    }
    else
    {
      //Check if Left Leg is support
      if (LFvnorm < VelocityThres)
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

      if (LHvnorm < VelocityThres)
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

      if (RFvnorm < VelocityThres)
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
      if (RHvnorm < VelocityThres)
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



      //Determine support
      if (contactL && contactR)
      {
        if (lf > rf)
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
      else if(contactLF && contactRF)
      ////////////
      else if (contactLF)
      {
        support_leg = "LFLeg";
        support_foot_frame = LFfoot_frame;
      }
      else if (contactLH)
      {
        support_leg = "LHLeg";
        support_foot_frame = LHfoot_frame;
      }
      else if (contactRF)
      {
        support_leg = "RFLeg";
        support_foot_frame = RFfoot_frame;
      }
      else if (contactRH)
      {
        support_leg = "RHLeg";
        support_foot_frame = RHfoot_frame;
      }
    }




    if(!firstContact)
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

  void SchmittTrigger(double lf, double rf)
  {
    contactL = 0;
    contactR = 0;

    p = rf + lf;
    pl = 0;
    pr = 0;

    if (p != 0)
    {
      pl = lf / p;
      pr = rf / p;
    }
    // Initial support leg
    if (firstContact)
    {
      if (lf > rf)
      {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      firstContact = false;
    }
    else
    {
      if (lf > LegHighThres && lf < StrikingContact)
      {
        contactL = 1;
      }
      if (lf < LegLowThres)
      {
        contactL = 0;
      }

      if (rf > LegHighThres && rf < StrikingContact)
      {
        contactR = 1;
      }

      if (rf < LegLowThres)
      {
        contactR = 0;
      }

      //Determine support
      if (contactL && contactR)
      {
        if (lf > rf)
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
      else if (contactL)
      {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else if (contactR)
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
    }
    if(!firstContact)
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
};
}