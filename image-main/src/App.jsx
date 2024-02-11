import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import PageNotFound from "./pages/PageNotFound";
import Login from "./pages/Login";
import { useState, useEffect } from "react";

import Signup from "./pages/Signup";
import Nav from "./components/Navb";

function App() {
  const [user, setUser] = useState("");

  return (
    <div className=" container ">
      <BrowserRouter>
        <Nav user={user} setUser={setUser} />
        <Routes>
          <Route index element={<Home user={user} />} />
          <Route path="login" element={<Login setUser={setUser} />} />
          <Route path="signup" element={<Signup />} />
          <Route path="*" element={<PageNotFound />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
