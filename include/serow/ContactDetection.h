#include <serow/Gaussian.h>
#include <string.h>


namespace serow{
class ContactDetection
{
private:
  Gaussian gs;
  double sigmalf, sigmarf;
  double sigmalc, sigmarc;
  double xmin, xmax, ymin, ymax;
  double lfmin, rfmin;
  double plf, prf, plc, prc;
  double pl, pr, p;
  double VelocityThres;
  bool firstContact;
  int contactL, contactR;

  //Smidtt Trigger
  double LegLowThres, LegHighThres, StrikingContact;
  std::string support_foot_frame, support_leg, lfoot_frame, rfoot_frame;
  
  double computeForceContactProb(double fmin, double sigma, double f)
  {
    return 1.000 - gs.cdf(fmin, f, sigma);
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

  double computeContactProb(double lf, double rf, double coplx, double coply, double coprx, double copry)
  {

    plf = computeForceContactProb(lfmin, sigmalf, lf);
    prf = computeForceContactProb(rfmin, sigmarf, rf);
    plc = computeCOPContactProb(xmax, xmin, sigmalc, coplx) * computeCOPContactProb(ymax, ymin, sigmalc, coply);
    prc = computeCOPContactProb(xmax, xmin, sigmarc, coprx) * computeCOPContactProb(ymax, ymin, sigmarc, copry);
    pr = prf * prc;
    pl = plf * plc;
    p = pl + pr;


    if (p != 0)
    {
      pl = pl / p;
      pr = pr / p;
    }
    else
    {
      pl = 0;
      pr = 0;
    }
    // cout<<"PROB IN CONTACT"<<endl;
    // cout<<pl<<endl;
    // cout<<pr<<endl;
    contactL=0;
    if(pl>=0.2)
      contactL=1;

    contactR=0;
    if(pr>=0.2)
      contactR=1;
    
  }

public:
  void init(std::string lfoot_frame_, std::string rfoot_frame_, double lfmin_, double rfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmalf_, double sigmarf_, double sigmalc_, double sigmarc_, double VelocityThres_)
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
    contactL = 0;
    contactR = 0;
    pl = 0;
    pr = 0;
    plf = 0;
    prf = 0; 
    plc = 0;
    prc = 0;
    VelocityThres = VelocityThres_;
    firstContact = true;
  }

  void init(std::string lfoot_frame_, std::string rfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_)
  {

    lfoot_frame = lfoot_frame_;
    rfoot_frame = rfoot_frame_;
    LegHighThres = LegHighThres_; 
    LegLowThres = LegLowThres_;  
    StrikingContact = StrikingContact_; 
    VelocityThres = VelocityThres_;
    firstContact = true;
    pl = 0;
    pr = 0;
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
    
    p = rf+lf;
    pl = 0;
    pr = 0;

    if(p != 0)
    {
      pl = lf/p;
      pr = rf/p;
    }

    if(firstContact)
    {
       if(lf>rf)
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
        if(pl>=0.2)
          contactL=1;

        contactR=0;
        if(pr>=0.2)
          contactR=1;
          
        firstContact = false;
    }
    else
    {
            //Check if Left Leg is support
            if (lvnorm<VelocityThres)
            {
                if ( lf > LegHighThres  && lf<StrikingContact)
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
            if (rvnorm<VelocityThres)
            {
                if ( rf > LegHighThres  && rf<StrikingContact)
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
            if(contactL && contactR)
            {
                if(lf>rf)
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
            else if(contactL)
            {
                support_leg = "LLeg";
                support_foot_frame = lfoot_frame;
            }
            else if(contactR)
            {
                support_leg = "RLeg";
                support_foot_frame = rfoot_frame;
            }
    }
    

  }





  void SchmittTrigger(double lf, double rf)
  {
    contactL = 0;
    contactR = 0;
    
    p=rf+lf;
    pl = 0;
    pr = 0;

    if(p != 0)
    {
      pl = lf/p;
      pr = rf/p;
    }
    // Initial support leg
    if(firstContact)
    {
       if(lf>rf)
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
                if ( lf > LegHighThres  && lf<StrikingContact)
                {
                    contactL = 1;
                }
                if (lf < LegLowThres)
                {
                    contactL = 0;
                }
           

           
                if ( rf > LegHighThres  && rf<StrikingContact)
                {
                    contactR = 1;
                }
            
                if (rf < LegLowThres)
                {
                    contactR = 0;
                }
            
            
            //Determine support
            if(contactL && contactR)
            {
                if(lf>rf)
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
            else if(contactL)
            {
                support_leg = "LLeg";
                support_foot_frame = lfoot_frame;
            }
            else if(contactR)
            {
                support_leg = "RLeg";
                support_foot_frame = rfoot_frame;
            }
    }
    

  }





  void computeSupportFoot(double lf, double rf, double coplx, double coply, double coprx, double copry, double lvnorm,double rvnorm)
  {

    computeContactProb(lf,  rf,  coplx,  coply,  coprx,  copry);
    //Choose Initial Support Foot based on Contact Force
    if(firstContact)
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
  }
};
}
