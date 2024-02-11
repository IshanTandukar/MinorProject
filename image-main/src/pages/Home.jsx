import Nav from "../components/Navb";
import Dandd from "../components/Dandd";
import { useEffect, useState } from "react";

function Home({ user }) {
  return (
    <div className="flex flex-col gap-10 ">
      {user == "" ? <h1>Login to Colorize Image</h1> : <Dandd />}
    </div>
  );
}

export default Home;
