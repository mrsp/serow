#include <serow/Gaussian.h>
#include <string.h>




namespace serow
{
class ContactDetection
{
private:
  Gaussian gs;
  //Probabilistic Kinematic - Contact Wrench detection
  double sigmalf, sigmarf;
  double sigmalc, sigmarc;
  double sigmalv, sigmarv;
  double xmin, xmax, ymin, ymax;
  double lfmin, rfmin;
  double plf, prf, plc, prc, plv, prv;
  double pl, pr, p;
  double VelocityThres;
  bool firstContact, useCOP, useKin;
  int contactL, contactR;
  double prob_TH;
  std::string support_foot_frame, support_leg, lfoot_frame, rfoot_frame;

  double deltalfz, deltarfz, lf_, rf_;
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

    if (p != 0)
    {
      pl = pl / p;
      pr = pr / p;
      pl += 0.5;
      pr += 0.5;
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
  }

public:
  void init(std::string lfoot_frame_, std::string rfoot_frame_, double lfmin_, double rfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmalf_, double sigmarf_, double sigmalc_, double sigmarc_, double VelocityThres_, double sigmalv_, double sigmarv_, bool useCOP_ = false, bool useKin_ = false, double prob_TH_ = 0.9)
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
    pl = 0;
    pr = 0;
    plf = 0;
    prf = 0;
    plc = 0;
    prc = 0;
    plv = 0;
    prv = 0;
    VelocityThres = VelocityThres_;
    prob_TH = prob_TH_;
    useCOP = useCOP_;
    useKin = useKin_;
    firstContact = true;
    deltalfz = 0;
    deltarfz = 0;

  }

  void init(std::string lfoot_frame_, std::string rfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_, double prob_TH_ = 0.9)
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

  }
  double getDiffForce()
  {
    cout<<"Delta FORCE"<<endl;
    cout<<fabs(deltalfz + deltarfz)<<endl;
    return fabs(deltalfz + deltarfz);
  }
  void computeSupportFoot(double lf, double rf, double coplx, double coply, double coprx, double copry, double lvnorm, double rvnorm)
  {

   
    computeContactProb(lf, rf, coplx, coply, coprx, copry, lvnorm, rvnorm);
    //Choose Initial Support Foot based on Contact Force
    if (firstContact)
    {
      if (pl > pr)
      {
        // Initial support leg
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
      if ((pl >= pr) && (lvnorm < VelocityThres && rvnorm > VelocityThres))
      {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else if ((pr >= pl) && (rvnorm < VelocityThres && lvnorm > VelocityThres))
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
    }

    if(!firstContact)
    {
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    lf_ = lf;
    rf_ = rf;
  }

  double getLLegContactProb()
  {
    return pl;
  }

  double getRLegContactProb()
  {
    return pr;
  }

  int isLLegContact()
  {
    return contactL;
  }

  int isRLegContact()
  {
    return contactR;
  }

  std::string getSupportFrame()
  {
    return support_foot_frame;
  }
  std::string getSupportLeg()
  {
    return support_leg;
  }

  void SchmittTriggerWithKinematics(double lf, double rf, double lvnorm, double rvnorm)
  {
    contactL = 0;
    contactR = 0;

    p = rf + lf;
    pl = 0;
    pr = 0;

    if (p != 0)
    {
      pl = lf / p + 0.5;
      pr = rf / p + 0.5;
    }

    if (firstContact)
    {
      if (lf > rf)
      {
        // Initial support leg
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      if (pl >= prob_TH)
        contactL = 1;

      contactR = 0;
      if (pr >= prob_TH)
        contactR = 1;

      firstContact = false;
    }
    else
    {
      //Check if Left Leg is support
      if (lvnorm < VelocityThres)
      {
        if (lf > LegHighThres && lf < StrikingContact)
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

      //Check if Right Leg is support
      if (rvnorm < VelocityThres)
      {
        if (rf > LegHighThres && rf < StrikingContact)
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
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    lf_ = lf;
    rf_ = rf;
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
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    lf_ = lf;
    rf_ = rf;
  }
};
}