import React, { useState, useEffect, useContext } from "react";
import Countdown from "react-countdown";
import images from "../assets";
import Image from "next/image";
import { Typography, Button } from '@mui/material';
import Head from 'next/head'

//INTERNAL IMPORT
import { VotingContext } from "../context/Voter";
import Style from "../styles/index.module.css";
import Card from "../components/card/card";
import { CustomButton } from "../components/Button/Button.jsx"


const index = () => {
  const {
    getNewCandidate,
    candidateArray,
    giveVote,
    checkIfWalletIsConnected,
    candidateLength,
    getAllVoterData,
    currentAccount,
    voterLength,
  } = useContext(VotingContext);

  useEffect(() => {
    checkIfWalletIsConnected();
  }, []);

  return (
    <>
      <div>
        <Head>
          <title>
            Decentralised E-Voting
          </title>
        </Head>
      </div>

      <div className={Style.homepage1}>
        {currentAccount && (
          <div className={Style.text1}>
            <Typography variant="h3" align="center" fontWeight={"Bold"}>
              Vote as per convenience with our
            </Typography>

            <Typography variant="h3" align="center" color={"#0096FF"} fontWeight={"Bold"}>
              Decentralised Voting App
            </Typography> <br></br>
            <Typography variant="subtitle1" align="center" style={{
              marginTop: 20,
              marginBottom: 30,
            }}>
              Our decentralized voting app puts the power of democracy in your hands.
              Say goodbye to concerns about fraud and manipulation. Our blockchain-based platform ensures the integrity of every vote. Easy to use and highly secure, our app enables you to participate in elections from anywhere. Join us in shaping the future of governance.
            </Typography>
            <Button variant="contained" color="primary" style={{
              margin: " 0 auto",
              display: "flex",
            }}>
              Get Started
            </Button>
          </div>


        )

        }
      </div >

      <div className={Style.home}>
        {currentAccount && (
          <div className={Style.winner}>
            <div className={Style.winner_info}>
              <div className={Style.candidate_list}>
                <p>
                  No. of Candidates:<span>{candidateLength}</span>
                </p>
              </div>
              <div className={Style.candidate_list}>
                <p>
                  No. of Voters:<span>{voterLength}</span>
                </p>
              </div>
            </div>
            <div className={Style.winner_message}>
              <small>
                <Countdown date={Date.now() + 1000000000} />
              </small>
            </div>
          </div>
        )
        }

        <Card candidateArray={candidateArray} giveVote={giveVote} />
      </div >


    </>
  );
};

export default index;
